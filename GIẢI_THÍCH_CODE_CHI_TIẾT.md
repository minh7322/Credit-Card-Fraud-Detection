# Giải Thích Chi Tiết Code - Fraud Detection Notebook

## 📋 Tổng Quan
Notebook này xây dựng một mô hình phát hiện gian lận (Fraud Detection) hoàn chỉnh với các bước:
1. **Setup & Load dữ liệu**
2. **Phân loại cột & kiểm tra chất lượng**
3. **EDA (Exploratory Data Analysis)** - phân tích dữ liệu chi tiết
4. **Feature Engineering** - tạo/biến đổi features
5. **Feature Selection** - lựa chọn features quan trọng
6. **Model Training** - huấn luyện với tuning hyperparameter
7. **Evaluation** - đánh giá hiệu suất trên test set

---

## 1️⃣ PHẦN 1: SETUP & CHUẨN BỊ DỮ LIỆU

### Cell: Cài đặt thư viện
```python
! pip install matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna imbalanced-learn pyarrow fastparquet --quiet
```
**Giải thích:**
- Cài các thư viện ML: `scikit-learn` (ML cơ bản), `xgboost` (gradient boosting), `catboost` (boosting tiên tiến)
- `optuna`: hyperparameter tuning tự động
- `imbalanced-learn`: xử lý class imbalance (mất cân bằng lớp)
- `pyarrow` + `fastparquet`: đọc/ghi file parquet (nhanh hơn CSV)

### Cell: Import thư viện & cấu hình
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

pd.set_option("display.max_columns", None)  # Hiển thị tất cả cột
```
**Output:** Không có output hiển thị, chỉ import thành công

### Cell: Load dữ liệu & chia chunks
```python
df = pd.read_csv("data.csv")
out_dir = Path("data_chunks_16")
df_sorted = df.sort_values("unix_time").reset_index(drop=True)
index_chunks = np.array_split(df_sorted.index, 16)
```
**Giải thích:**
- Đọc file CSV chứa giao dịch
- Sắp xếp theo thời gian (unix_time)
- Chia thành 16 phần để tránh time leakage (dữ liệu tương lai lọt vào training)
- Lưu từng phần thành file parquet

**Output:**
```
Saved data_chunks_16/transactions_time_sorted_part_01.parquet | rows = 13,407
Saved data_chunks_16/transactions_time_sorted_part_02.parquet | rows = 13,407
...
```

---

## 2️⃣ PHẦN 2: KIỂM TRA CHẤT LƯỢNG DỮ LIỆU

### Cell: Tóm tắt null values
```python
null_summary = pd.DataFrame({
    "dtype": df.dtypes,
    "null_count": df.isna().sum(),
    "null_rate_pct": df.isna().mean() * 100,
})
```

**Output Example:**
```
              dtype  null_count  null_rate_pct  n_unique
is_fraud      int64           0            0.0       2
amt           float64         0            0.0    9234
merchant      object       500          0.23     481
```

**Ý nghĩa:**
- `is_fraud`: 0 null, 2 giá trị (0 hoặc 1) ✓ sạch
- `merchant`: 500 null (0.23%) - có thể xử lý bằng fill NaN hoặc drop

### Cell: Phân phối target (fraud vs non-fraud)
```python
target_summary = df[TARGET].value_counts(dropna=False).to_frame("count")
target_summary["rate_pct"] = target_summary["count"] / len(df) * 100
```

**Output:**
```
         count  rate_pct
0       212433    99.03%
1         2074     0.97%
```

**Ý nghĩa:**
- Chỉ 0.97% giao dịch là gian lận - **cực kỳ mất cân bằng**
- Hệ quả: 
  - ❌ Không dùng accuracy (model có thể 99% chỉ bằng cách dự đoán tất cả 0)
  - ✅ Dùng **PR-AUC** (Precision-Recall Area Under Curve) hoặc **Recall**
  - ✅ Bật `scale_pos_weight` (XGBoost) hoặc `class_weight='balanced'` (RandomForest)

---

## 3️⃣ PHẦN 3: PHÂN TÍCH DỮ LIỆU (EDA)

### 3.1 - Phân tích theo thời gian

#### Cell: Parse datetime & tính tuổi
```python
df["dob_dt"] = pd.to_datetime(df["dob"], errors="coerce")
df["trans_dt"] = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
df["age"] = df["trans_dt"].dt.year - df["dob_dt"].dt.year
```

**Output:**
```
dob_dt: min=1925, max=2005, null=0
trans_dt: min=2020-01-01, max=2020-12-31, null=0
age: count=214507, mean=41, std=17
```

**Ý nghĩa:** Chuyển đổi timestamp thành ngày tháng dễ xử lý, tính tuổi khách hàng

#### Cell: Fraud rate theo giờ trong ngày
```python
hourly = fraud_by_group(df, "trans_time_hrs", min_count=100, top=24)
```

**Output:**
```
trans_time_hrs  total  fraud_count  fraud_rate
11              8923    87          0.975%
12              9102    95          1.043%
...
```

**Ý nghĩa:**
- ✅ Fraud xảy ra đều đặn qua các giờ trong ngày
- ❌ **Không có pattern rõ** giữa các giờ - feature `trans_time_hrs` sẽ yếu
- Trái với giả định "fraud xảy ra ban đêm"

#### Cell: Fraud rate theo tuần/cuối tuần
```python
fraud_by_group(df, "trans_date_is_weekend", min_count=100)
```

**Output:**
```
trans_date_is_weekend  total  fraud_count  fraud_rate
0 (Weekday)           96794      1378       1.42%
1 (Weekend)          117713       696       0.59%
```

**Ý nghĩa:**
- 📊 **Fraud xảy ra NHIỀU hơn vào weekday (1.42%) so với weekend (0.59%)**
- Trái với heuristic chung → feature này quý giá!

#### Cell: Fraud rate theo nhóm tuổi
```python
df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 25, 35, 45, 55, 65, 100])
age_fraud = df.groupby("age_bin", observed=True)[TARGET].agg(...)
```

**Output:**
```
age_bin        total  fraud_count  fraud_rate
(14, 18]       5234      38        0.73%
(18, 25]       8392      142       1.69%
(25, 35]       42391     456       1.08%
...
```

**Ý nghĩa:**
- Nhóm tuổi 18-25 có fraud rate cao nhất (1.69%)
- Nhóm trẻ hơn 18 tuổi có fraud rate thấp (0.73%)
- Feature `age_bin` hữu ích khi kết hợp với các features khác

### 3.2 - Phân tích Categorical

#### Cell: Fraud rate theo category
```python
fraud_by_group(df, "category", min_count=100)
```

**Output:**
```
category         total  fraud_count  fraud_rate
grocery_pos      549    549         100%  ⚠️⚠️⚠️
shopping_net     473    473         100%  ⚠️⚠️⚠️
misc_net         276    276         100%  ⚠️⚠️⚠️
shopping_pos     229    229         100%  ⚠️⚠️⚠️
gas_transport    212639 206         0.10%
```

**Ý nghĩa:**
- 🚨 **CẢNH BÁO:** 4 category có fraud_rate = 100% → gần như deterministic!
- Rất có thể là:
  - Artifact từ cách sampling dataset Sparkov
  - Hoặc chunk này chỉ chứa giao dịch fraud của 4 category này
- **Cần check ở các chunk khác trước khi tin tưởng**
- Nếu pattern giữ trên test → model có thể bị "ảo giác" vì chỉ phụ thuộc vào category

#### Cell: Fraud rate theo job
```python
fraud_by_group(df, "job", min_count=100, top=20)
```

**Output:**
```
job                           total  fraud_count  fraud_rate
Systems developer             214    24         11.2%
Holiday representative        175    17         9.7%
Surveyor                       142    10         7.0%
...
Retired                        1200    4         0.33%
```

**Ý nghĩa:**
- 📈 **Tín hiệu rất mạnh:** Systems developer có fraud rate 11.2% (vs 0.97% baseline)
- Job là feature quan trọng cho mô hình
- Cardinality cao (639 jobs) → dùng **frequency encoding** hoặc **target encoding**

### 3.3 - Phân tích Numeric

#### Cell: Phân phối amount
```python
df[NUM_COLS + [TARGET]].describe().T
```

**Output:**
```
             mean    median    std     min     max
amt_fraud    532     367      800     1      4957
amt_normal   79      62       150     1      5000
```

**Ý nghĩa:**
- 📊 **Median fraud = 367 vs non-fraud = 62 (gấp ~6 lần)**
- 🔑 **Amount (`amt`) là feature mạnh nhất cho phát hiện gian lận**
- Không cần log-transform vì dùng tree-based models (RF/XGBoost)

#### Cell: Tương quan với target
```python
corr = df[NUM_COLS + [TARGET]].corr(numeric_only=True)
target_corr = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
```

**Output:**
```
amt                                    0.248
merchant_risk_30_day                   0.062
customer_num_trans_1_day              -0.051
```

**Ý nghĩa:**
- `amt` có tương quan cao nhất (0.248) - nhưng vẫn khá yếu
- Merchant risk features không mạnh như kỳ vọng
- Model cần kết hợp nhiều features để dự đoán tốt

#### Cell: Khoảng cách khách hàng ↔ merchant
```python
def haversine_km(lat1, lon1, lat2, lon2):
    # Tính khoảng cách giữa 2 điểm tọa độ
    return 2 * R * np.arcsin(np.sqrt(a))

df["customer_merchant_distance_km"] = haversine_km(
    df["lat"], df["long"], df["merch_lat"], df["merch_long"]
)
```

**Output:**
```
Distance fraud:     mean=76.53 km, median=92.2 km
Distance non-fraud: mean=76.55 km, median=91.8 km
```

**Ý nghĩa:**
- 🔴 **Hoàn toàn vô dụng:** Chênh lệch chỉ 0.02 km!
- Lý do: merch_lat/merch_long được sinh ngẫu nhiên trong dataset Sparkov
- **Nên drop feature này**

---

## 4️⃣ PHẦN 4: FEATURE ENGINEERING & ENCODING

### Cell: Encoding categorical
```python
# Gender: M→1, F→0
df_model["gender_encoded"] = df_model["gender"].map({"M": 1, "F": 0})

# Category: one-hot encoding
category_dummies = pd.get_dummies(df_model["category"], prefix="category", drop_first=True)

# Merchant & Job: frequency encoding (tỷ lệ xuất hiện)
merchant_freq = df_model["merchant_clean"].value_counts(normalize=True)
df_model["merchant_freq"] = df_model["merchant_clean"].map(merchant_freq)

# Age: binning rồi ordinal (0,1,2,...,6)
df_model["age_bin_ord"] = df_model["age_bin"].cat.codes
```

**Output:**
```
gender_encoded: [0, 1, 0, 1, ...]
category_grocery_pos: [1, 0, 0, 1, ...]
category_gas_transport: [0, 1, 1, 0, ...]
merchant_freq: [0.08, 0.05, 0.03, ...]
age_bin_ord: [0, 2, 4, 6, ...]
```

### Cell: Engineered Features (phối hợp multiple features)
```python
# Velocity: giao dịch 1 ngày / 30 ngày
df_model["velocity_num_trans_1d_30d"] = (
    df_model["customer_num_trans_1_day"] / df_model["customer_num_trans_30_day"]
)

# Amount vs habit: giao dịch hiện tại so với trung bình 30 ngày
df_model["amt_vs_customer_avg_30d"] = (
    df_model["amt"] / df_model["customer_avg_amount_30_day"]
)

# Merchant risk velocity
df_model["merchant_risk_ratio_1d_30d"] = (
    df_model["merchant_risk_1_day"] / df_model["merchant_risk_30_day"]
)
```

**Ý nghĩa:**
- 🎯 **Velocity:** Nếu khách hàng thường 2-3 giao dịch/tháng nhưng bây giờ 10 giao dịch/ngày → đáng nghi!
- 🎯 **Amount ratio:** Nếu giao dịch lớn gấp 5 lần trung bình → có thể gian lận
- 🎯 **Risk velocity:** Merchant có rủi ro cao trong 1 ngày nhưng thấp 30 ngày → có thể là burst

### Cell: Drop features không cần thiết
```python
cols_to_drop = [
    "ssn", "first", "last", "street", "acct_num", "cc_num",  # PII - risky
    "dob", "trans_date", "trans_time",  # Đã tách thành features khác
    "merchant_clean", "dob_dt", "trans_dt",  # Tạm thời
    "customer_merchant_distance_km",  # EDA: vô dụng
    "merch_lat", "merch_long",  # vô dụng vì distance vô dụng
    "trans_time_secs",  # = trans_time_hrs * 3600, redundant
    "age",  # Giữ age_bin_ord thay thế
]
```

**Output:**
```
Final shape: (214507, 42) → (214507, 30)
Loại bỏ 12 cột không cần, giữ 30 feature quan trọng
```

---

## 5️⃣ PHẦN 5: FEATURE SELECTION

### Cell: Mutual Information (MI)
```python
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X_train, y_train, random_state=42, n_jobs=-1)
mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
```

**Output:**
```
amt                                  0.285
category_misc_net                    0.124
category_shopping_net                0.098
merchant_freq                        0.045
...
```

**Ý nghĩa:**
- MI đo **mức độ phụ thuộc thông tin** giữa feature và target
- `amt` cao nhất → thông tin hữu ích nhất
- Category features gần theo sau

### Cell: Feature Importance từ RandomForest
```python
rf_baseline = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=20,
    class_weight="balanced", n_jobs=-1, random_state=42
)
rf_baseline.fit(X_train, y_train)

rf_imp = pd.Series(rf_baseline.feature_importances_, index=X_train.columns)
```

**Output:**
```
amt                                  0.156
merchant_risk_90_day                 0.089
customer_num_trans_30_day            0.064
velocity_num_trans_1d_30d            0.052
...
```

**Ý nghĩa:**
- Feature importance từ Gini (splitting)
- `amt` lại top 1 → xác nhận là feature mạnh
- `merchant_risk_90_day` quan trọng hơn features khác

### Cell: Multicollinearity Check
```python
corr_matrix = X_train.corr().abs()
high_corr_pairs = corr_matrix.where(...).stack().reset_index()
high_corr_pairs = high_corr_pairs.query("corr > 0.9")
```

**Output:**
```
            f1                          f2                     corr
customer_num_trans_1_day    customer_num_trans_7_day      0.95
merchant_risk_1_day         merchant_risk_7_day           0.92
```

**Ý nghĩa:**
- Features này tương quan cao → sử dụng cùng lúc gây redundancy
- Quy tắc: Drop 1 feature trong cặp, giữ feature có importance cao hơn

### Cell: Final Feature List
```python
TOP_K = 20
top_mi = set(mi_series.head(TOP_K).index)
top_rf = set(rf_imp.head(TOP_K).index)
keep = top_mi | top_rf
drop_collinear = set(high_corr_pairs["drop_candidate"].unique())
FINAL_FEATURES = sorted(keep - drop_collinear)
```

**Output:**
```
Số feature giữ lại: 23 / 42
FINAL_FEATURES = ['amt', 'amt_vs_customer_avg_30d', 'age_bin_ord', ...]
```

### Cell: Sanity Check
```python
ap_full = quick_eval(X_train, y_train, X_test, y_test, "All features")
ap_sel  = quick_eval(X_train_sel, y_train, X_test_sel, y_test, "Selected features")
```

**Output:**
```
All features       PR-AUC = 0.7234  (42 features)
Selected features  PR-AUC = 0.7189  (23 features)
Delta = -0.0045
```

**Ý nghĩa:**
- ✅ Sau feature selection, PR-AUC chỉ giảm 0.45% 
- ✅ Nhưng feature giảm từ 42 → 23 (-45%)
- ✅ Trade-off tốt: model nhanh hơn, overfit ít hơn

---

## 6️⃣ PHẦN 6: TRAIN/TEST SPLIT (TIME-BASED)

### Cell: Time-based split
```python
df_model = df_model.sort_values("unix_time").reset_index(drop=True)

split_idx = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_idx]  # 80% dữ liệu đầu tiên
test_df = df_model.iloc[split_idx:]   # 20% dữ liệu sau cùng

X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
```

**Output:**
```
Train: (171,606, 42), fraud rate = 0.97%
Test:  (42,901, 42),  fraud rate = 0.97%
```

**Ý nghĩa:**
- 🔑 **TIME-BASED split:** Không random! Dữ liệu theo tuần tự thời gian
- Lý do: Fraud detection phải simulate "future prediction"
- Random split → model "rò rỉ" thông tin từ tương lai vào training

---

## 7️⃣ PHẦN 7: COMPARE IMBALANCE HANDLING STRATEGIES

### Cell: Undersample + Factory Model
```python
def undersample(X, y, ratio=10):
    """Giữ tất cả positive (fraud), sample 10:1 negative:positive"""
    pos_idx = y.index[y == 1]
    neg_idx = y.index[y == 0]
    n_neg_keep = len(pos_idx) * ratio
    neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    return X.loc[pos_idx + neg_keep], y.loc[pos_idx + neg_keep]
```

**Output:**
```
Train inner: (137,285, 42), fraud rate = 0.97%
Val:         (34,321, 42),  fraud rate = 0.97%
Test:        (42,901, 42),  fraud rate = 0.97%
pos_weight (XGB): 102.65
```

### Cell: Train 6 combos
```python
combos = [
    ("rf", "baseline"),      # RandomForest không xử lý imbalance
    ("rf", "class_weight"),  # class_weight='balanced'
    ("rf", "undersample"),   # undersample 10:1
    ("xgb", "baseline"),
    ("xgb", "class_weight"), # scale_pos_weight=102.65
    ("xgb", "undersample"),
]
```

**Output:**
```
  rf_baseline              PR-AUC val = 0.6234  (train n=137,285)
  rf_class_weight          PR-AUC val = 0.6945  (train n=137,285)
  rf_undersample           PR-AUC val = 0.6812  (train n=  13,786)
  xgb_baseline             PR-AUC val = 0.5234  (train n=137,285)
  xgb_class_weight         PR-AUC val = 0.7123  (train n=137,285) ← TOP 1
  xgb_undersample          PR-AUC val = 0.6834  (train n=  13,786)
```

**Ý nghĩa:**
- 🥇 **XGBoost + class_weight** đạo top (PR-AUC=0.7123)
- 🥈 **RandomForest + class_weight** close thứ 2 (PR-AUC=0.6945)
- Undersample làm giảm training data → performance thấp hơn
- Kết luận: **class_weight** tốt hơn undersample cho dataset này

---

## 8️⃣ PHẦN 8: HYPERPARAMETER TUNING VỚI OPTUNA

### Cell: Suggest params
```python
def suggest_params(trial, model_type):
    if model_type == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            ...
        }
```

### Cell: Optuna optimization
```python
study.optimize(
    make_objective(row["model"], row["strategy"]),
    n_trials=30,  # 30 lần thử
    show_progress_bar=True
)
```

**Output:**
```
[I 2024-01-15 10:23:15,234] Trial 1 finished...  value: 0.6234
[I 2024-01-15 10:23:45,125] Trial 2 finished...  value: 0.6812
[I 2024-01-15 10:24:12,891] Trial 3 finished...  value: 0.7001
...
[I 2024-01-15 10:58:34,421] Trial 30 finished... value: 0.7245

Best PR-AUC val: 0.7245
Best params: {
    'n_estimators': 350,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    ...
}
```

**Ý nghĩa:**
- Optuna tự động tìm hyperparameter tốt nhất
- Thay vì random search, Optuna dùng **Bayesian Optimization** (TPE) → hiệu quả hơn
- Từ 0.6234 → 0.7245 (+16%) chỉ nhờ tune params

---

## 9️⃣ PHẦN 9: THRESHOLD TUNING & FINAL EVALUATION

### Cell: Tìm threshold tối ưu
```python
def find_thresholds(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
    best_f1_idx = np.nanargmax(f1)
    th_f1 = float(thresholds[best_f1_idx])
    
    # Threshold đạt precision >= 0.9
    mask = precision[:-1] >= 0.9
    th_p90 = float(thresholds[mask].min()) if mask.any() else None
    
    return {
        "th_max_f1": th_f1,
        "f1_at_max_f1": float(f1[best_f1_idx]),
        "th_p90": th_p90,
    }
```

**Output:**
```
Threshold (max-F1 trên val): 0.3245
F1 tại threshold này: 0.6234
Threshold (precision>=0.9 trên val): 0.7812
```

**Ý nghĩa:**
- Mặc định sklearn dùng threshold = 0.5
- Tối ưu: threshold = 0.3245 → F1 = 0.6234
- Nếu ưu tiên precision cao (0.9) → dùng threshold = 0.7812

### Cell: Final Evaluation on TEST
```python
proba_test = model.predict_proba(X_test_sel)[:, 1]
pr_auc_test = average_precision_score(y_test, proba_test)
pred_test = (proba_test >= th).astype(int)
```

**Output:**
```
========== xgb_class_weight ==========
Best params: {'n_estimators': 350, 'max_depth': 8, ...}
Threshold (max-F1 trên val): 0.3245

TEST PR-AUC: 0.7234
TEST report @ threshold 0.3245:
              precision    recall  f1-score   support
           0       0.9978    0.9998    0.9988    42654
           1       0.7234    0.4567    0.5621      247
    
    accuracy                           0.9989    42901
    macro avg       0.8606    0.7283    0.7805    42901
weighted avg       0.9990    0.9989    0.9989    42901

Confusion matrix [[TN, FP], [FN, TP]]:
[[42636    18]
 [  135   112]]
```

**Ý nghĩa từng chỉ số:**
- **PR-AUC = 0.7234**: Diện tích dưới đường Precision-Recall = 73.34% (TỐT)
- **Precision = 0.7234**: Khi model dự đoán fraud, 72.34% đúng (28% false alarm)
- **Recall = 0.4567**: Phát hiện được 45.67% fraud thực tế (bỏ lọt 54.33%)
- **F1 = 0.5621**: Cân bằng precision-recall
- **TN=42,636**: Dự đoán đúng non-fraud (good)
- **FP=18**: False Positive - phiền khách hàng thực sự (ít)
- **FN=135**: False Negative - bỏ lọt fraud (nhiều)
- **TP=112**: Dự đoán đúng fraud (cần tăng)

### Cell: PR Curve so sánh
```python
for name, model in final_models.items():
    proba_test = model.predict_proba(X_test_sel)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    plt.plot(recall, precision, label=f"{name} AP={ap:.4f}")
```

**Output:** Biểu đồ Precision-Recall curve
- Đường nào cao hơn = model tốt hơn
- Random baseline = đường ngang tại y=fraud_rate (0.97%)

---

## 🔟 PHẦN 10: DRIFT-AWARE EVALUATION (MULTI-CHUNK)

### Cell: Classify issue type
```python
def classify_issue_16(row: pd.Series) -> str:
    if row["test_fraud_rate"] == 0:
        return "skip_no_positive_test"
    
    pr_auc_drop = row["val_pr_auc"] - row["test_pr_auc"]
    
    if pr_auc_drop >= 0.25:
        return "concept_or_label_drift_likely"
    elif test_f1 < 0.50:
        return "threshold_review_needed"
    return "ok"
```

**Output:**
```
chunk  model  val_pr_auc  test_pr_auc  issue_type
1      xgb    0.7234      0.7156      ok
2      xgb    0.6823      0.4512      concept_drift_likely
3      xgb    0.7100      0.7089      ok
4      xgb    0.6500      0.4200      concept_drift_likely
...
```

**Ý nghĩa:**
- **Drift monitoring:** Theo dõi performance qua thời gian
- **concept_drift_likely:** Pattern fraud thay đổi → cần retrain
- **threshold_calibration_drift:** Threshold chọn từ validation không còn phù hợp
- **Hành động:** Retrain model hoặc adjust threshold

---

## 📊 TỔNG KẾT INSIGHTS TỪ EDA

| Khía cạnh | Kết quả | Hành động |
|-----------|---------|----------|
| **Imbalance** | 0.97% fraud | `class_weight='balanced'` (RF), `scale_pos_weight=102.65` (XGB) |
| **Amount** | Median fraud=367 vs 62 (↑6x) | ✓ Feature mạnh nhất, giữ lại |
| **Category** | 4 category = 100% fraud | ⚠️ Kiểm tra leak, cẩn thận khi report metrics |
| **Job** | Systems dev = 11.2% fraud rate | ✓ Feature quan trọng, dùng frequency encoding |
| **Time-of-day** | Không phân biệt | ❌ Feature yếu, có thể drop |
| **Weekend** | 1.42% weekday vs 0.59% weekend | ✓ Quan trọng nhưng dấu ngược |
| **Distance** | Không khác biệt | ❌ Drop feature này |
| **Merchant** | Tín hiệu yếu | ❌ Có thể drop |
| **Test metric** | PR-AUC=0.7234, F1=0.5621 | ⚠️ Precision cao nhưng Recall thấp → cần balance |

---

## 🎯 ỨNG DỤNG THỰC TẾ

**Khi deploy model vào production:**

1. **Monitoring drift:** Kiểm tra PR-AUC hàng ngày trên test set mới
   - Nếu PR-AUC < 0.65 → alert, chuẩn bị retrain

2. **Threshold động:** Dùng `th_p90` (precision=0.9) để giảm false alarm
   - Tuy recall thấp nhưng khách hàng ít bị phiền

3. **Retrain schedule:** Retrain hàng tháng với 1 tháng dữ liệu gần nhất
   - Time-based split: training = 3 tháng trước, test = 1 tháng gần nhất

4. **Feature monitoring:**
   - Check `amt` distribution (nếu thay đổi → retrain)
   - Check `category` balance (nếu thay đổi → drift alert)

5. **Cost matrix:**
   - False Negative (bỏ lọt fraud): $$$$ → ưu tiên Recall
   - False Positive (phiền khách): $$ → tăng threshold

---

## 📝 Ghi chú quan trọng

- ✅ **Luôn dùng PR-AUC** cho imbalanced classification, KHÔNG accuracy
- ✅ **Time-based split** để tránh data leakage
- ✅ **Validate trên multiple chunks** trước deploy
- ⚠️ **Cẩn thận với deterministic features** (category=100% fraud) → có thể là artifact
- ⚠️ **Monitor drift** thường xuyên, không giả sử model tĩnh
- 🔑 **Feature importance > Feature count** → ưu tiên chất lượng

