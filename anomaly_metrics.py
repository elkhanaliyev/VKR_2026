from datetime import datetime
import pandas as pd
import numpy as np
import os
import joblib
from scipy.spatial.distance import mahalanobis

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             average_precision_score)

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated_with_anomalies.xlsx"
OUTPUT_FILE = "power_data_anomalies_stat.xlsx"
ANOMALY_START_DAY = 61

EPS = 1.5
MIN_SAMPLES = 3

random_state = 42
np.random.seed(random_state)
START_DATETIME = datetime(2025, 10, 5, 0, 2)

print("=" * 70)
print("Z-score + Махаланобис")
print("=" * 70)

# ====================== ЗАГРУЗКА ДАННЫХ ======================
df = pd.read_excel(INPUT_FILE)

df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

features = ['Реактивная мощность', 'Выходной коэффициент мощности',
            'Полная мощность', 'Ток']

df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df = df.sort_values('Time').reset_index(drop=True)

# Определяем аномалии
if 'Is_Anomaly' in df.columns:
    df['Is_Anomaly'] = df['Is_Anomaly'].map({
        'Да': True, 'Нет': False,
        'Yes': True, 'No': False,
        True: True, False: False
    }).fillna(df['day_num'] >= ANOMALY_START_DAY)
else:
    df['Is_Anomaly'] = df['day_num'] >= ANOMALY_START_DAY

normal_df = df[~df['Is_Anomaly']].copy()
anomaly_df = df[df['Is_Anomaly']].copy()

print(f"\nВсего записей: {len(df):,}")
print(f"Нормальных (дни 1–60): {len(normal_df):,}")
print(f"Аномальных (с дня {ANOMALY_START_DAY}): {len(anomaly_df):,}")

# ====================== StandardScaler ======================
print("\n" + "=" * 70)
print("StandardScaler")
print("=" * 70)

scaler = StandardScaler()
train_data = normal_df[features].dropna()
scaler.fit(train_data)

X_all = scaler.transform(df[features].dropna())
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
df['DBSCAN_Label'] = dbscan.fit_predict(X_all)

print(f"EPS = {EPS}, MIN_SAMPLES = {MIN_SAMPLES}")
print(f"Кластеров найдено: {df['DBSCAN_Label'].nunique() - (1 if -1 in df['DBSCAN_Label'].values else 0)}")
print(f"Точек-шума (-1): {(df['DBSCAN_Label'] == -1).sum():,}")

# ====================== ТАБЛИЦА Z-SCORE ======================
print("\n" + "=" * 70)
print("ТАБЛИЦА 8 – Статистическая оценка выраженности аномалий (Z-score)")
print("=" * 70)

normal_mean = normal_df[features].mean()
normal_std = normal_df[features].std()

z_table_rows = []
for feat in features:
    anom_vals = anomaly_df[feat].dropna()
    z_scores = (anom_vals - normal_mean[feat]) / normal_std[feat]
    
    z_table_rows.append({
        'Канал': feat,
        'Среднее (аномалии)': round(anom_vals.mean(), 4),
        'Стд. откл. (аномалии)': round(anom_vals.std(), 4),
        'Мин': round(anom_vals.min(), 2),
        'Макс': round(anom_vals.max(), 3),
        'Средний z-score': round(z_scores.mean(), 3),
        'Мин z-score': round(z_scores.min(), 2),
        'Макс z-score': round(z_scores.max(), 3),
        'Доля |z| ≥ 1.5 (%)': round((z_scores.abs() >= 1.5).mean() * 100, 1)
    })

z_df = pd.DataFrame(z_table_rows)
print(z_df.to_string(index=False))

# ====================== ТАБЛИЦА МАХАЛАНОБИС ======================
print("\n" + "=" * 70)
print("ТАБЛИЦА 9 – Статистика расстояния Махаланобиса для аномалий")
print("=" * 70)

normal_X = normal_df[features].dropna().values
anomaly_X = anomaly_df[features].dropna().values

mean_vec = normal_X.mean(axis=0)
cov_matrix = np.cov(normal_X, rowvar=False)
cov_inv = np.linalg.pinv(cov_matrix)

mahal_dists = []
for x in anomaly_X:
    diff = x - mean_vec
    d = np.sqrt(diff @ cov_inv @ diff)
    mahal_dists.append(d)

mahal_arr = np.array(mahal_dists)

mahal_results = [
    ['Количество аномальных точек', len(mahal_arr)],
    ['Среднее Mahalanobis', round(mahal_arr.mean(), 3)],
    ['Медиана Mahalanobis', round(np.median(mahal_arr), 3)],
    ['Минимум', round(mahal_arr.min(), 3)],
    ['Максимум', round(mahal_arr.max(), 3)],
    ['Стандартное отклонение', round(mahal_arr.std(), 3)],
    ['Доля ≥ 3.0', f"{(mahal_arr >= 3.0).mean() * 100:.1f} %"],
    ['Доля ≥ 3.5', f"{(mahal_arr >= 3.5).mean() * 100:.1f} %"],
    ['Доля ≥ 4.0', f"{(mahal_arr >= 4.0).mean() * 100:.1f} %"],
    ['Доля ≥ 5.0', f"{(mahal_arr >= 5.0).mean() * 100:.1f} %"]
]

mahal_df = pd.DataFrame(mahal_results, columns=['Показатель', 'Значение'])
print(mahal_df.to_string(index=False))

# ====================== ДОБАВЛЕНИЕ MAHALANOBIS В DF ======================
# Создаём словарь для быстрого поиска
anomaly_indices = anomaly_df[features].dropna().index
mahal_dict = dict(zip(anomaly_indices, mahal_dists))
df['Mahalanobis'] = df.index.map(mahal_dict)

# ====================== СОХРАНЕНИЕ ======================
print("\n" + "=" * 70)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 70)

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='DBSCAN_Results', index=False)
    z_df.to_excel(writer, sheet_name='Z_Score_Table', index=False)
    mahal_df.to_excel(writer, sheet_name='Mahalanobis_Table', index=False)

print(f"✅ Итоговый файл сохранён: {OUTPUT_FILE}")
print(f"   Лист 'DBSCAN_Results' – данные с метками кластеров и Mahalanobis")
print(f"   Лист 'Z_Score_Table' – таблица 8")
print(f"   Лист 'Mahalanobis_Table' – таблица 9")