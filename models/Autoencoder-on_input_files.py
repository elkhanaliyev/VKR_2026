from datetime import datetime
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated_with_anomalies.xlsx"      # ← измени на свой файл
OUTPUT_FILE = "power_data_AE_4features_predictions.xlsx"
MODEL_PATH = "models/autoencoder_4features_model.pkl"

ANOMALY_START_DAY = 61
START_DATETIME = datetime(2025, 10, 5, 0, 2)

print("=== Autoencoder (4 features) — Inference ===\n")

# ====================== ЗАГРУЗКА МОДЕЛИ ======================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

model_info = joblib.load(MODEL_PATH)

autoencoder = model_info['autoencoder']
scaler = model_info['scaler']
threshold = model_info['threshold']
features = model_info['features']

print("Autoencoder (4 features) загружен успешно")
print(f"Порог реконструкции: {threshold:.6f}\n")

# ====================== ЗАГРУЗКА НОВОГО ФАЙЛА ======================
df = pd.read_excel(INPUT_FILE)
df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=features).reset_index(drop=True)
df = df.sort_values('Time').reset_index(drop=True)

print(f"Загружено записей: {len(df):,}")

# ====================== ПРИМЕНЕНИЕ МОДЕЛИ ======================
print("Вычисляем ошибки реконструкции...")

X_full = df[features].values.astype(np.float32)
X_full_scaled = scaler.transform(X_full).astype(np.float32)

full_recon = autoencoder.predict(X_full_scaled, verbose=0)
full_mse = np.mean((X_full_scaled - full_recon) ** 2, axis=1)

df["AE_recon_mse"] = full_mse
df["Is_AE_Anomaly"] = (full_mse > threshold).astype(int)
df["anomaly_score"] = full_mse

# ====================== МЕТРИКИ (если есть ground truth) ======================
if 'Is_Anomaly' in df.columns:
    df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
    df['y_pred'] = df['Is_AE_Anomaly']
   
    mask_period = df['day_num'] >= ANOMALY_START_DAY
   
    print("\n" + "="*75)
    print("МЕТРИКИ ПРЕДСКАЗАНИЯ")
    print("="*75)
   
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    score = df['anomaly_score'].values
   
    print(f"F1-score     : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Precision    : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall       : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"MCC          : {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"Balanced Acc : {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"AUROC        : {roc_auc_score(y_true, score):.4f}")
    print(f"PR-AUC       : {average_precision_score(y_true, score):.4f}")
   
    print(f"\n--- Период аномалий (день {ANOMALY_START_DAY}+) ---")
    y_true_p = df.loc[mask_period, 'y_true'].values
    y_pred_p = df.loc[mask_period, 'y_pred'].values
    score_p = df.loc[mask_period, 'anomaly_score'].values
   
    print(f"F1-score     : {f1_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"Recall       : {recall_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"AUROC        : {roc_auc_score(y_true_p, score_p):.4f}")

# ====================== СТАТИСТИКА ======================
print("\n" + "="*65)
print("СТАТИСТИКА ОБНАРУЖЕНИЯ")
print("="*65)

total_anom = df['Is_AE_Anomaly'].sum()
period_anom = df.loc[df['day_num'] >= ANOMALY_START_DAY, 'Is_AE_Anomaly'].sum()

print(f"Всего найдено аномалий: {total_anom} ({total_anom/len(df)*100:.3f}%)")
print(f"Аномалий в периоде {ANOMALY_START_DAY}+ дней: {period_anom}")

if 'y_true' in df.columns:
    true_period = df.loc[df['day_num'] >= ANOMALY_START_DAY, 'y_true'].sum()
    print(f"Истинных аномалий в периоде: {true_period}")

# ====================== СОХРАНЕНИЕ ======================
df.to_excel(OUTPUT_FILE, index=False)
print(f"\n✅ Предсказания успешно сохранены в файл:\n{OUTPUT_FILE}")