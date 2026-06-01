from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier


TARGET = "is_fraud"
CHUNK_DIR = Path("data_chunks_16")
OUT_DIR = Path("chunk_model_results_16")
OUT_DIR.mkdir(exist_ok=True)

FINAL_FEATURES = [
    "age_bin_ord",
    "amt",
    "amt_vs_customer_avg_30d",
    "category_grocery_pos",
    "category_misc_net",
    "category_shopping_net",
    "category_shopping_pos",
    "city_pop",
    "customer_avg_amount_30_day",
    "customer_avg_amount_7_day",
    "customer_avg_amout_1_day",
    "customer_num_trans_30_day",
    "customer_num_trans_7_day",
    "gender_encoded",
    "job_freq",
    "lat",
    "long",
    "merchant_freq",
    "merchant_num_trans_30_day",
    "merchant_risk_1_day",
    "merchant_risk_30_day",
    "merchant_risk_7_day",
    "trans_date_is_weekend",
    "trans_time_day",
    "trans_time_is_night",
]

PARAMS = {
    "xgb_undersample": {
        "strategy": "undersample",
        "params": {
            "n_estimators": 150,
            "max_depth": 8,
            "learning_rate": 0.016664,
            "subsample": 0.948150,
            "colsample_bytree": 0.781684,
            "min_child_weight": 6,
            "reg_alpha": 0.189588,
            "reg_lambda": 0.007976,
            "gamma": 0.660841,
        },
    },
    "xgb_class_weight": {
        "strategy": "class_weight",
        "params": {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.015384,
            "subsample": 0.880095,
            "colsample_bytree": 0.754347,
            "min_child_weight": 1,
            "reg_alpha": 0.155126,
            "reg_lambda": 0.005929,
            "gamma": 3.799895,
        },
    },
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dob_dt"] = pd.to_datetime(df["dob"], errors="coerce")
    df["trans_dt"] = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
    df["age"] = df["trans_dt"].dt.year - df["dob_dt"].dt.year
    df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 25, 35, 45, 55, 65, 100])

    df["gender_encoded"] = df["gender"].map({"M": 1, "F": 0}).fillna(-1)

    category_dummies = pd.get_dummies(df["category"], prefix="category", drop_first=True)
    df = pd.concat([df, category_dummies], axis=1)

    df["merchant_clean"] = df["merchant"].str.replace("^fraud_", "", regex=True)
    df["merchant_freq"] = df["merchant_clean"].map(
        df["merchant_clean"].value_counts(normalize=True)
    )
    df["job_freq"] = df["job"].map(df["job"].value_counts(normalize=True))
    df["age_bin_ord"] = df["age_bin"].cat.codes

    df["velocity_num_trans_1d_30d"] = (
        df["customer_num_trans_1_day"] / df["customer_num_trans_30_day"]
    ).replace([np.inf, -np.inf], 0).fillna(0)
    df["amt_vs_customer_avg_30d"] = (
        df["amt"] / df["customer_avg_amount_30_day"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0)
    df["merchant_risk_ratio_1d_30d"] = (
        df["merchant_risk_1_day"] / df["merchant_risk_30_day"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0)

    return df.reindex(columns=FINAL_FEATURES, fill_value=0).astype(float)


def undersample(X: pd.DataFrame, y: pd.Series, ratio: int = 10, random_state: int = 42):
    pos_idx = y.index[y == 1]
    neg_idx = y.index[y == 0]
    n_neg_keep = min(len(pos_idx) * ratio, len(neg_idx))
    rng = np.random.RandomState(random_state)
    neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    keep = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(keep)
    return X.loc[keep], y.loc[keep]


def find_thresholds(y_true: pd.Series, proba: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return {
            "th_max_f1": 0.5,
            "f1_at_max_f1": 0.0,
            "precision_at_max_f1": 0.0,
            "recall_at_max_f1": 0.0,
        }

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_f1_idx = int(np.nanargmax(f1))
    return {
        "th_max_f1": float(thresholds[best_f1_idx]),
        "f1_at_max_f1": float(f1[best_f1_idx]),
        "precision_at_max_f1": float(precision[best_f1_idx]),
        "recall_at_max_f1": float(recall[best_f1_idx]),
    }


def average_precision_safe(y_true: pd.Series, proba: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(average_precision_score(y_true, proba))


def make_xgb(params: dict, scale_pos_weight: float | None = None) -> XGBClassifier:
    defaults = {
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "aucpr",
        "tree_method": "hist",
    }
    defaults.update(params)
    if scale_pos_weight is not None:
        defaults["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**defaults)


def run_one_chunk(chunk_no: int) -> list[dict]:
    path = CHUNK_DIR / f"transactions_time_sorted_part_{chunk_no:02d}.parquet"
    df = pd.read_parquet(path).sort_values("unix_time").reset_index(drop=True)
    X = build_features(df)
    y = df[TARGET].astype(int)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    val_split_idx = int(len(X_train) * 0.8)
    X_tr_inner = X_train.iloc[:val_split_idx]
    y_tr_inner = y_train.iloc[:val_split_idx]
    X_val = X_train.iloc[val_split_idx:]
    y_val = y_train.iloc[val_split_idx:]

    pos_weight = (y_tr_inner == 0).sum() / max((y_tr_inner == 1).sum(), 1)

    rows = []
    for name, spec in PARAMS.items():
        X_fit, y_fit = X_tr_inner, y_tr_inner
        scale_pos_weight = None

        if spec["strategy"] == "undersample":
            X_fit, y_fit = undersample(X_tr_inner, y_tr_inner)
        elif spec["strategy"] == "class_weight":
            scale_pos_weight = pos_weight

        model = make_xgb(spec["params"], scale_pos_weight=scale_pos_weight)
        model.fit(X_fit, y_fit)

        proba_val = model.predict_proba(X_val)[:, 1]
        th_info = find_thresholds(y_val, proba_val)
        threshold = th_info["th_max_f1"]

        proba_test = model.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= threshold).astype(int)
        cm = confusion_matrix(y_test, pred_test, labels=[0, 1])

        rows.append(
            {
                "chunk": chunk_no,
                "model": name,
                "rows": len(df),
                "train_inner_size": len(X_fit),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "train_inner_fraud_rate": float(y_fit.mean()),
                "val_fraud_rate": float(y_val.mean()),
                "test_fraud_rate": float(y_test.mean()),
                "val_pr_auc": average_precision_safe(y_val, proba_val),
                "test_pr_auc": average_precision_safe(y_test, proba_test),
                "threshold_max_f1_val": threshold,
                "val_f1_at_threshold": th_info["f1_at_max_f1"],
                "test_f1": float(f1_score(y_test, pred_test, zero_division=0)),
                "test_TN": int(cm[0, 0]),
                "test_FP": int(cm[0, 1]),
                "test_FN": int(cm[1, 0]),
                "test_TP": int(cm[1, 1]),
            }
        )
    return rows


def plot_results(results: pd.DataFrame) -> None:
    pivot_pr = results.pivot(index="chunk", columns="model", values="test_pr_auc")
    pivot_f1 = results.pivot(index="chunk", columns="model", values="test_f1")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    pivot_pr.plot(ax=axes[0], marker="o")
    axes[0].set_title("XGBoost test PR-AUC by chunk (data_chunks_16)")
    axes[0].set_ylabel("PR-AUC")
    axes[0].grid(True, alpha=0.3)

    pivot_f1.plot(ax=axes[1], marker="o")
    axes[1].set_title("XGBoost test F1 by chunk (data_chunks_16)")
    axes[1].set_xlabel("Chunk")
    axes[1].set_ylabel("F1")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "xgb_chunks_02_16_metrics.png", dpi=150)
    plt.close(fig)


def main() -> None:
    all_rows = []
    for chunk_no in range(2, 17):
        print(f"Running chunk {chunk_no:02d}...")
        rows = run_one_chunk(chunk_no)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(
            OUT_DIR / "xgb_chunks_02_16_results_partial.csv", index=False
        )
        for row in rows:
            print(
                f"  {row['model']}: val_PR-AUC={row['val_pr_auc']:.4f}, "
                f"test_PR-AUC={row['test_pr_auc']:.4f}, "
                f"test_F1={row['test_f1']:.4f}, "
                f"threshold={row['threshold_max_f1_val']:.4f}"
            )

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT_DIR / "xgb_chunks_02_16_results.csv", index=False)
    plot_results(results)
    print(f"Saved {OUT_DIR / 'xgb_chunks_02_16_results.csv'}")
    print(f"Saved {OUT_DIR / 'xgb_chunks_02_16_metrics.png'}")


if __name__ == "__main__":
    main()
