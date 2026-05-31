from datetime import datetime
import pandas as pd
import numpy as np
import os
import time
import psutil
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated_with_anomalies_s3.xlsx"
OUTPUT_FILE = "power_data_IsolationForest_8features.xlsx"
ANOMALY_START_DAY = 61
START_DATETIME = datetime(2025, 10, 5, 0, 2)

# Параметры IsolationForest
N_ESTIMATORS = 200
MAX_SAMPLES = "auto"
CONTAMINATION = 0.02
RANDOM_STATE = 42
N_JOBS = -1

USE_SCORE_THRESHOLD = True
SCORE_THRESHOLD_QUANTILE = 0.99

print("=== Isolation Forest на 8 признаках (уровни + дельты) ===\n")

# ====================== ЗАГРУЗКА ДАННЫХ ======================
df = pd.read_excel(INPUT_FILE)
df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

level_cols = ['Реактивная мощность', 'Выходной коэффициент мощности',
              'Полная мощность', 'Ток']
df[level_cols] = df[level_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=["Time"] + level_cols).reset_index(drop=True)
df = df.sort_values('Time').reset_index(drop=True)

print(f"Всего записей: {len(df):,}")
print(f"Дней 1–60 (обучение): {len(df[df['day_num'] <= 60]):,}")

# ====================== СОЗДАНИЕ 8 ПРИЗНАКОВ ======================
df['Q'] = df['Реактивная мощность']
df['cos_phi'] = df['Выходной коэффициент мощности']
df['S'] = df['Полная мощность']
df['I'] = df['Ток']

df['delta_Q'] = df['Q'].diff().fillna(0)
df['delta_cos'] = df['cos_phi'].diff().fillna(0)
df['delta_S'] = df['S'].diff().fillna(0)
df['delta_I'] = df['I'].diff().fillna(0)

features_level = ['Q', 'cos_phi', 'S', 'I']
features_delta = ['delta_Q', 'delta_cos', 'delta_S', 'delta_I']

print("8 признаков созданы (уровни + дельты)")

# ====================== ОБУЧЕНИЕ ======================
train_mask = df['day_num'] <= 60

scaler_level = StandardScaler()
scaler_delta = StandardScaler()

X_level_train = df.loc[train_mask, features_level].values
X_delta_train = df.loc[train_mask, features_delta].values

X_level_scaled = scaler_level.fit_transform(X_level_train)
X_delta_scaled = scaler_delta.fit_transform(X_delta_train)

X_train_scaled = np.hstack([X_level_scaled, X_delta_scaled])

print(f"Обучаем IsolationForest (n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION}) на 8 признаках...")
iso = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples=MAX_SAMPLES,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)
iso.fit(X_train_scaled)
print("Обучение завершено.")

# ====================== ПРИМЕНЕНИЕ НА ВСЕ ДАННЫЕ ======================
X_level_full = df[features_level].values
X_delta_full = df[features_delta].values

X_level_scaled = scaler_level.transform(X_level_full)
X_delta_scaled = scaler_delta.transform(X_delta_full)
X_full_scaled = np.hstack([X_level_scaled, X_delta_scaled])

decision = iso.decision_function(X_full_scaled)
anomaly_score = -decision

df['IF_decision'] = decision
df['IF_anomaly_score'] = anomaly_score

if USE_SCORE_THRESHOLD:
    train_scores = anomaly_score[train_mask.values]
    score_threshold = float(np.quantile(train_scores, SCORE_THRESHOLD_QUANTILE))
    df['Is_IF_Anomaly'] = (df['IF_anomaly_score'] > score_threshold).astype(int)
    print(f"Используется порог по {SCORE_THRESHOLD_QUANTILE:.0%}-квантилю: {score_threshold:.6f}")
else:
    pred_labels = iso.predict(X_full_scaled)
    df['Is_IF_Anomaly'] = (pred_labels == -1).astype(int)

# ====================== МЕТКИ ======================
df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
df['y_pred'] = df['Is_IF_Anomaly']

# ====================== ФУНКЦИИ МЕТРИК ======================
def compute_all_metrics(y_true, y_pred, anomaly_score=None):
    metrics = {}
    metrics['f1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
   
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics['confusion_matrix'] = cm
    metrics['far'] = fp / (fp + tn + 1e-12)
   
    if anomaly_score is not None and len(np.unique(y_true)) > 1:
        metrics['auroc'] = roc_auc_score(y_true, anomaly_score)
        metrics['prauc'] = average_precision_score(y_true, anomaly_score)
        metrics['score_source'] = 'anomaly_score'
    else:
        metrics['auroc'] = np.nan
        metrics['prauc'] = np.nan
        metrics['score_source'] = 'N/A'
    return metrics

def print_metrics(metrics, set_name=""):
    print("\n" + "="*65)
    if set_name:
        print(f"РЕЗУЛЬТАТЫ: {set_name}")
    print("="*65)
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall : {metrics['recall']:.4f}")
    print(f"FAR : {metrics['far']:.4f}")
    print(f"MCC : {metrics['mcc']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_acc']:.4f}")
    print(f"AUROC ({metrics['score_source']}): {metrics['auroc']:.4f}")
    print(f"PR-AUC ({metrics['score_source']}): {metrics['prauc']:.4f}")
   
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"Normal  {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"Anomaly {cm[1,0]:6d} {cm[1,1]:6d}")
    print("="*65)

# ====================== ВЫВОД РЕЗУЛЬТАТОВ ======================
print("\n" + "="*65)
print("РЕЗУЛЬТАТЫ ISOLATION FOREST НА 8 ПРИЗНАКАХ")
print("="*65)

mask_period = df['day_num'] >= ANOMALY_START_DAY

metrics_all = compute_all_metrics(df['y_true'], df['y_pred'], df['IF_anomaly_score'])
print_metrics(metrics_all, "ВСЕ ДАННЫЕ")

metrics_period = compute_all_metrics(
    df.loc[mask_period, 'y_true'],
    df.loc[mask_period, 'y_pred'],
    df.loc[mask_period, 'IF_anomaly_score']
)
print_metrics(metrics_period, f"ПЕРИОД АНОМАЛИЙ (день {ANOMALY_START_DAY}+)")

# ====================== СРАВНИТЕЛЬНЫЕ МЕТРИКИ ======================
print("\n" + "="*75)
print("СРАВНИТЕЛЬНЫЕ МЕТРИКИ ISOLATION FOREST (8 features)")
print("="*75)

# --- Training time ---
start_train = time.perf_counter()

# Повторяем обучение для точного замера
X_level_train = df.loc[train_mask, features_level].values
X_delta_train = df.loc[train_mask, features_delta].values
X_level_scaled = scaler_level.fit_transform(X_level_train)
X_delta_scaled = scaler_delta.fit_transform(X_delta_train)
X_train_scaled = np.hstack([X_level_scaled, X_delta_scaled])

iso.fit(X_train_scaled)

train_time = time.perf_counter() - start_train
print(f"Training time  : {train_time:.4f} секунд")

# --- Inference time ---
start_inf = time.perf_counter()

X_level_full = df[features_level].values
X_delta_full = df[features_delta].values
X_level_scaled = scaler_level.transform(X_level_full)
X_delta_scaled = scaler_delta.transform(X_delta_full)
X_full_scaled = np.hstack([X_level_scaled, X_delta_scaled])

decision = iso.decision_function(X_full_scaled)
anomaly_score = -decision

inference_time = time.perf_counter() - start_inf
print(f"Inference time : {inference_time:.4f} секунд (на всех данных)")

# --- Peak RAM ---
process = psutil.Process(os.getpid())
ram_mb = process.memory_info().rss / (1024 * 1024)

print(f"Peak RAM (cRAM) : {ram_mb:.1f} МБ")

# --- Готовая строка для таблицы ---
print("\n" + "-"*90)
print("ГОТОВАЯ СТРОКА ДЛЯ ТАБЛИЦЫ СРАВНЕНИЯ:")
print(f"IsolationForest (8 features)    | {train_time:.3f} s     | {inference_time:.3f} s     | {ram_mb:.0f} МБ")
print("-"*90)

# ====================== СОХРАНЕНИЕ МОДЕЛИ ======================
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

model_info = {
    'model': iso,
    'scaler_level': scaler_level,
    'scaler_delta': scaler_delta,
    'features_level': features_level,
    'features_delta': features_delta,
    'N_ESTIMATORS': N_ESTIMATORS,
    'CONTAMINATION': CONTAMINATION,
    'USE_SCORE_THRESHOLD': USE_SCORE_THRESHOLD,
    'SCORE_THRESHOLD': score_threshold if USE_SCORE_THRESHOLD else None,
    'SCORE_THRESHOLD_QUANTILE': SCORE_THRESHOLD_QUANTILE,
    'start_datetime': START_DATETIME,
    'trained_on_days': 60,
    'model_type': 'IsolationForest_8features'
}

model_path = f"{MODEL_DIR}isolationforest_8features_model.pkl"
joblib.dump(model_info, model_path)
print(f"\nМодель успешно сохранена: {model_path}")

# ====================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ======================
df.to_excel(OUTPUT_FILE, index=False)
print(f"Результат успешно сохранён в: {OUTPUT_FILE}")