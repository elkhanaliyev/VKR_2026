from datetime import datetime
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.neighbors import NearestNeighbors

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated_with_anomalies.xlsx"
OUTPUT_FILE = "power_data_DBSCAN_4features_predictions.xlsx"
MODEL_PATH = "models/dbscan_4features_model.pkl"

ANOMALY_START_DAY = 61
START_DATETIME = datetime(2025, 10, 5, 0, 2)

print("=== Применение сохранённой DBSCAN модели (4 признака) ===\n")

# ====================== ЗАГРУЗКА МОДЕЛИ ======================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

model_info = joblib.load(MODEL_PATH)

dbscan = model_info['dbscan']
scaler = model_info['scaler']           # для 4-features модели используется один scaler
EPS = model_info['EPS']
MIN_SAMPLES = model_info['MIN_SAMPLES']
features = model_info['features']

print(f"Модель загружена: DBSCAN 4 features")
print(f"EPS = {EPS}, MIN_SAMPLES = {MIN_SAMPLES}")
print(f"Используемые признаки: {features}")

# ====================== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ======================
df = pd.read_excel(INPUT_FILE)

df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

# Приводим числовые столбцы
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df = df.sort_values('Time').reset_index(drop=True)

print(f"Загружено записей: {len(df):,}")

# ====================== ПРИМЕНЕНИЕ МОДЕЛИ ======================
print("Применяем модель к данным...")

X_full = df[features].values
X_full_scaled = scaler.transform(X_full)

# Используем NearestNeighbors по core-точкам модели
core_samples = dbscan.components_
nn = NearestNeighbors(radius=EPS, n_jobs=-1)
nn.fit(core_samples)

distances, _ = nn.radius_neighbors(X_full_scaled, return_distance=True)
neighbor_counts = np.array([len(d) for d in distances])

# Предсказания
df['DBSCAN_neighbor_count'] = neighbor_counts
df['Is_DBSCAN_Anomaly'] = (neighbor_counts < MIN_SAMPLES).astype(int)

# Anomaly score
max_neighbors = neighbor_counts.max() if len(neighbor_counts) > 0 else 1
df['anomaly_score'] = 1 - (neighbor_counts / (max_neighbors + 1e-12))
df['anomaly_score'] = np.clip(df['anomaly_score'], 0, 1)

# ====================== МЕТРИКИ (если есть истинные метки) ======================
if 'Is_Anomaly' in df.columns:
    df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
    df['y_pred'] = df['Is_DBSCAN_Anomaly']
    
    mask_period = df['day_num'] >= ANOMALY_START_DAY
    
    print("\n" + "="*70)
    print("МЕТРИКИ ПРЕДСКАЗАНИЯ")
    print("="*70)
    
    # Импортируем метрики только если они нужны
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 matthews_corrcoef, balanced_accuracy_score,
                                 roc_auc_score, average_precision_score,
                                 confusion_matrix)
    
    # Метрики за весь период
    y_true = df['y_true']
    y_pred = df['y_pred']
    score = df['anomaly_score']
    
    print(f"F1-score     : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Precision    : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall       : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"MCC          : {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"Balanced Acc : {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"AUROC        : {roc_auc_score(y_true, score):.4f}")
    print(f"PR-AUC       : {average_precision_score(y_true, score):.4f}")
    
    # Метрики только с 61 дня
    print(f"\n--- Только период аномалий (день {ANOMALY_START_DAY}+) ---")
    y_true_p = df.loc[mask_period, 'y_true']
    y_pred_p = df.loc[mask_period, 'y_pred']
    score_p = df.loc[mask_period, 'anomaly_score']
    
    print(f"F1-score     : {f1_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"Precision    : {precision_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"Recall       : {recall_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"AUROC        : {roc_auc_score(y_true_p, score_p):.4f}")

# ====================== ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ======================
print("\n" + "="*65)
print("СТАТИСТИКА ОБНАРУЖЕНИЯ")
print("="*65)
total_anom = df['Is_DBSCAN_Anomaly'].sum()
period_anom = df.loc[df['day_num'] >= ANOMALY_START_DAY, 'Is_DBSCAN_Anomaly'].sum()

print(f"Всего найдено аномалий: {total_anom} ({total_anom/len(df)*100:.3f}%)")
print(f"Аномалий в периоде {ANOMALY_START_DAY}+ дней: {period_anom}")

if 'y_true' in df.columns:
    true_in_period = df.loc[df['day_num'] >= ANOMALY_START_DAY, 'y_true'].sum()
    print(f"Истинных аномалий в периоде: {true_in_period}")

# ====================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ======================
df.to_excel(OUTPUT_FILE, index=False)
print(f"\nПредсказания успешно сохранены в файл:\n{OUTPUT_FILE}")