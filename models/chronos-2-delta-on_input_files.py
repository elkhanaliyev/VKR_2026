from datetime import datetime
import pandas as pd
import numpy as np
import os
import torch
import joblib
from tqdm import tqdm
from chronos import Chronos2Pipeline
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score, average_precision_score)

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated_with_anomalies_s3.xlsx"      # ← твой файл с дрейфом
OUTPUT_FILE = "power_data_CHRONOS2_8features_predictions.xlsx"
MODEL_PATH = "models/chronos2_8features_model.pkl"

ANOMALY_START_DAY = 61
START_DATETIME = datetime(2025, 10, 5, 0, 2)

CONTEXT_LEN = 288
PRED_LEN = 24
BATCH_SIZE = 32

THRESHOLD_MULTIPLIER = 1          # ← можно подкрутить (1.1 - 1.6)

print("=== Chronos-2 8 features — Inference ===\n")

# ====================== ЗАГРУЗКА МОДЕЛИ ======================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

model_info = joblib.load(MODEL_PATH)
scaler_level = model_info['scaler_level']
scaler_delta = model_info['scaler_delta']
level_features = model_info['level_features']
delta_features = model_info['delta_features']
best_cfg = model_info['best_cfg']

print(f"Модель Chronos-2 (8 признаков) загружена")
print(f"Агрегация: {best_cfg.get('aggregation')}, Percentile: {best_cfg.get('percentile')}")

# ====================== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ======================
df = pd.read_excel(INPUT_FILE)
df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

# Создаём 8 признаков (надёжно)
df['Q'] = df.get('Реактивная мощность', df.get('Q'))
df['cos_phi'] = df.get('Выходной коэффициент мощности', df.get('cos_phi'))
df['S'] = df.get('Полная мощность', df.get('S'))
df['I'] = df.get('Ток', df.get('I'))

df['delta_Q'] = df['Q'].diff().fillna(0)
df['delta_cos'] = df['cos_phi'].diff().fillna(0)
df['delta_S'] = df['S'].diff().fillna(0)
df['delta_I'] = df['I'].diff().fillna(0)

# ====================== МАСШТАБИРОВАНИЕ ======================
X_level = df[level_features].values
X_delta = df[delta_features].values

X_level_scaled = scaler_level.transform(X_level)
X_delta_scaled = scaler_delta.transform(X_delta)
X_full_scaled = np.hstack([X_level_scaled, X_delta_scaled])

print(f"Данные подготовлены. Записей: {len(df):,}")

# ====================== ЗАГРУЗКА CHRONOS-2 ======================
print("\nЗагружаем Chronos-2...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
    dtype=torch.float32,
)
print("Chronos-2 загружен.")

# ====================== ФУНКЦИЯ ИНФЕРЕНСА ======================
def get_forecast_errors(X_scaled, context_len=CONTEXT_LEN, pred_len=PRED_LEN):
    N, C = X_scaled.shape
    error_sum = np.zeros((N, C), dtype=np.float64)
    error_count = np.zeros(N, dtype=np.float64)
    
    step = pred_len
    starts = list(range(context_len, N - pred_len + 1, step))
    
    print(f"Выполняется {len(starts)} окон прогноза...")
    
    for i in tqdm(range(0, len(starts), BATCH_SIZE), desc="Chronos-2"):
        batch_starts = starts[i:i + BATCH_SIZE]
        frames = []
        
        for b_idx, s in enumerate(batch_starts):
            for ch in range(C):
                window = X_scaled[s - context_len:s, ch]
                ts = pd.date_range("2000-01-01", periods=context_len, freq="1min")
                frames.append(pd.DataFrame({
                    "id": f"ts_{b_idx}_{ch}",
                    "timestamp": ts,
                    "target": window.astype(np.float32)
                }))
        
        combined_df = pd.concat(frames, ignore_index=True)
        
        pred_df = pipeline.predict_df(
            combined_df,
            prediction_length=pred_len,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target"
        )
        
        for b_idx, s in enumerate(batch_starts):
            for ch in range(C):
                ts_id = f"ts_{b_idx}_{ch}"
                actual = X_scaled[s:s + pred_len, ch]
                
                # Извлечение медианы
                sub = pred_df[pred_df["id"] == ts_id]
                if 'quantile_level' in sub.columns:          # long format
                    med = sub[sub['quantile_level'] == 0.5]['target'].values
                    if len(med) == 0:
                        med = sub[sub['quantile_level'] == '0.5']['target'].values
                else:                                        # wide format
                    key = 0.5 if 0.5 in sub.columns else '0.5'
                    med = sub[key].values
                
                if len(med) == 0:
                    continue
                    
                n = min(len(actual), len(med))
                mae = np.abs(actual[:n] - med[:n])
                error_sum[s:s + n, ch] += mae
                error_count[s:s + n] += 1
    
    error_count = np.maximum(error_count, 1)
    errors = error_sum / error_count[:, np.newaxis]
    
    # Заполняем начало
    errors[:context_len] = errors[context_len:context_len+500].mean(axis=0) if len(errors) > context_len + 500 else errors.mean(axis=0)
    
    return errors


print("\nВычисляем ошибки прогнозирования...")
forecast_errors = get_forecast_errors(X_full_scaled)

# ====================== ПРИМЕНЕНИЕ ЛУЧШЕЙ КОНФИГУРАЦИИ ======================
if best_cfg.get('weights') is not None:
    errors = (forecast_errors * best_cfg['weights']).sum(axis=1)
else:
    errors = forecast_errors.max(axis=1)

df['forecast_error'] = errors
df['anomaly_score'] = np.clip(errors / errors.max(), 0, 1)

final_threshold = best_cfg['threshold'] * THRESHOLD_MULTIPLIER
df['Is_CHRONOS2_Anomaly'] = (errors > final_threshold).astype(int)
df['y_pred'] = df['Is_CHRONOS2_Anomaly']

# ====================== МЕТРИКИ ======================
if 'Is_Anomaly' in df.columns:
    df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
    mask_period = df['day_num'] >= ANOMALY_START_DAY
    
    print("\n" + "="*75)
    print("РЕЗУЛЬТАТЫ INFERENCE Chronos-2 (8 признаков)")
    print("="*75)
    
    y_true_p = df.loc[mask_period, 'y_true']
    y_pred_p = df.loc[mask_period, 'y_pred']
    score_p = df.loc[mask_period, 'anomaly_score']
    
    print(f"F1-score     : {f1_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"Precision    : {precision_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"Recall       : {recall_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"AUROC        : {roc_auc_score(y_true_p, score_p):.4f}")

# ====================== СОХРАНЕНИЕ ======================
df.to_excel(OUTPUT_FILE, index=False)
print(f"\nПредсказания сохранены в: {OUTPUT_FILE}")
print(f"Использован multiplier порога: {THRESHOLD_MULTIPLIER}")