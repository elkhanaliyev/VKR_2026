from datetime import datetime
import pandas as pd
import numpy as np
import os
import joblib

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated_with_anomalies.xlsx"
OUTPUT_FILE = "power_data_IsolationForest_4features_predictions.xlsx"
MODEL_PATH = "models/isolationforest_4features_model.pkl"

ANOMALY_START_DAY = 61
START_DATETIME = datetime(2025, 10, 5, 0, 2)

print("=== Применение сохранённой модели Isolation Forest (4 признака) ===\n")

# ====================== ЗАГРУЗКА МОДЕЛИ ======================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

model_info = joblib.load(MODEL_PATH)

iso = model_info['model']
scaler = model_info['scaler']
features = model_info['features']
USE_SCORE_THRESHOLD = model_info.get('USE_SCORE_THRESHOLD', True)
score_threshold = model_info.get('SCORE_THRESHOLD')

print(f"Модель загружена: IsolationForest 4 features")
print(f"n_estimators = {model_info.get('N_ESTIMATORS')}, contamination = {model_info.get('CONTAMINATION')}")

# ====================== ЗАГРУЗКА ДАННЫХ ======================
df = pd.read_excel(INPUT_FILE)
df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df = df.sort_values('Time').reset_index(drop=True)

print(f"Загружено записей: {len(df):,}")

# ====================== ПРИМЕНЕНИЕ МОДЕЛИ ======================
print("Применяем Isolation Forest к данным...")

X_full = df[features].values
X_full_scaled = scaler.transform(X_full)

decision = iso.decision_function(X_full_scaled)
anomaly_score = -decision

df['IF_decision'] = decision
df['IF_anomaly_score'] = anomaly_score

if USE_SCORE_THRESHOLD and score_threshold is not None:
    df['Is_IF_Anomaly'] = (df['IF_anomaly_score'] > score_threshold).astype(int)
    print(f"Применён сохранённый порог: {score_threshold:.6f}")
else:
    pred_labels = iso.predict(X_full_scaled)
    df['Is_IF_Anomaly'] = (pred_labels == -1).astype(int)

# ====================== МЕТРИКИ ======================
if 'Is_Anomaly' in df.columns:
    df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
    df['y_pred'] = df['Is_IF_Anomaly']
    
    mask_period = df['day_num'] >= ANOMALY_START_DAY
    
    print("\n" + "="*70)
    print("МЕТРИКИ ПРЕДСКАЗАНИЯ")
    print("="*70)
    
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 matthews_corrcoef, balanced_accuracy_score,
                                 roc_auc_score, average_precision_score,
                                 confusion_matrix)
    
    y_true = df['y_true']
    y_pred = df['y_pred']
    score = df['IF_anomaly_score']
    
    print(f"F1-score     : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Precision    : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall       : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"MCC          : {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"Balanced Acc : {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"AUROC        : {roc_auc_score(y_true, score):.4f}")
    print(f"PR-AUC       : {average_precision_score(y_true, score):.4f}")
    
    print(f"\n--- Период аномалий (день {ANOMALY_START_DAY}+) ---")
    y_true_p = df.loc[mask_period, 'y_true']
    y_pred_p = df.loc[mask_period, 'y_pred']
    score_p = df.loc[mask_period, 'IF_anomaly_score']
    
    print(f"F1-score     : {f1_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"Precision    : {precision_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"Recall       : {recall_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"AUROC        : {roc_auc_score(y_true_p, score_p):.4f}")

# ====================== СТАТИСТИКА ======================
print("\n" + "="*65)
print("СТАТИСТИКА ОБНАРУЖЕНИЯ")
print("="*65)
total_anom = df['Is_IF_Anomaly'].sum()
period_anom = df.loc[df['day_num'] >= ANOMALY_START_DAY, 'Is_IF_Anomaly'].sum()
print(f"Всего найдено аномалий: {total_anom} ({total_anom/len(df)*100:.3f}%)")
print(f"Аномалий в периоде {ANOMALY_START_DAY}+: {period_anom}")

# ====================== СОХРАНЕНИЕ ======================
df.to_excel(OUTPUT_FILE, index=False)
print(f"\nПредсказания успешно сохранены в: {OUTPUT_FILE}")