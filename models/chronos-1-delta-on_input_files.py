from datetime import datetime
import pandas as pd
import numpy as np
import os
import torch
import joblib
import pickle
from tqdm import tqdm
from chronos import BaseChronosPipeline
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated_with_anomalies_s3.xlsx"   # ← замени на свой файл
OUTPUT_FILE = "power_data_CHRONOS_8features_predictions.xlsx"
MODEL_PATH = "models/chronos_8features_model.pkl"

ANOMALY_START_DAY = 61
START_DATETIME = datetime(2025, 10, 5, 0, 2)

# Параметры инференса (должны совпадать с обучением)
CONTEXT_LEN = 864
PRED_LEN = 12
BATCH_SIZE = 32

print("=== Применение сохранённой модели Chronos 8 features ===\n")

# ====================== ЗАГРУЗКА МОДЕЛИ ======================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

model_info = joblib.load(MODEL_PATH)

scaler_level = model_info['scaler_level']
scaler_delta = model_info['scaler_delta']
level_features = model_info['level_features']
delta_features = model_info['delta_features']
best_cfg = model_info['best_cfg']
CONTEXT_LEN = model_info.get('CONTEXT_LEN', CONTEXT_LEN)
PRED_LEN = model_info.get('PRED_LEN', PRED_LEN)

print(f"Модель Chronos 8 features успешно загружена")
print(f"Лучшая конфигурация: {best_cfg['aggregation']} | percentile={best_cfg['percentile']}")

# ====================== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ======================
df = pd.read_excel(INPUT_FILE)
df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

# Создаём 8 признаков
df['Q'] = df['Реактивная мощность']
df['cos_phi'] = df['Выходной коэффициент мощности']
df['S'] = df['Полная мощность']
df['I'] = df['Ток']

df['delta_Q'] = df['Q'].diff().fillna(0)
df['delta_cos'] = df['cos_phi'].diff().fillna(0)
df['delta_S'] = df['S'].diff().fillna(0)
df['delta_I'] = df['I'].diff().fillna(0)

# Масштабирование
X_level = df[level_features].values
X_delta = df[delta_features].values

X_level_scaled = scaler_level.transform(X_level)
X_delta_scaled = scaler_delta.transform(X_delta)
X_full_scaled = np.hstack([X_level_scaled, X_delta_scaled])  # [N, 8]

print(f"Данные подготовлены. Всего записей: {len(df):,}")

# ====================== ЗАГРУЗКА CHRONOS ======================
print("\nЗагружаем Chronos-Bolt для инференса...")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",
    device_map="cpu",
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
)
print("Chronos-Bolt загружен.")

# ====================== ИНФЕРЕНС ======================
def get_forecast_errors(X_scaled, context_len=CONTEXT_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE):
    N, C = X_scaled.shape
    error_sum = np.zeros((N, C), dtype=np.float64)
    error_count = np.zeros(N, dtype=np.float64)
    
    step = pred_len
    starts = list(range(context_len, N - pred_len + 1, step))
    
    print(f"Выполняется {len(starts)} прогнозов...")
    
    for ch in range(C):
        ch_series = X_scaled[:, ch]
        for i in tqdm(range(0, len(starts), batch_size), desc=f"Канал {ch+1}/8"):
            batch_starts = starts[i:i + batch_size]
            contexts = [torch.tensor(ch_series[s - context_len:s], dtype=torch.float32)
                        for s in batch_starts]
            
            with torch.no_grad():
                forecast = pipeline.predict(torch.stack(contexts), prediction_length=pred_len)
            
            median_forecast = forecast.median(dim=1).values.numpy()
            
            for j, s in enumerate(batch_starts):
                actual = ch_series[s:s + pred_len]
                pred = median_forecast[j]
                mae = np.abs(actual - pred)
                error_sum[s:s + pred_len, ch] += mae
                error_count[s:s + pred_len] += 1
    
    error_count = np.maximum(error_count, 1)
    errors = error_sum / error_count[:, np.newaxis]
    
    # Заполняем начало (первые context_len точек)
    train_mean_err = errors[:1000].mean(axis=0)   # приближение
    errors[:context_len] = train_mean_err
    
    return errors


print("\nВычисляем ошибки прогнозирования...")
forecast_errors = get_forecast_errors(X_full_scaled)

# ====================== ПРИМЕНЕНИЕ ЛУЧШЕЙ КОНФИГУРАЦИИ ======================
if best_cfg['aggregation'] == 'weighted' and best_cfg['weights'] is not None:
    errors = (forecast_errors * best_cfg['weights']).sum(axis=1)
elif best_cfg['aggregation'] == 'mean':
    errors = forecast_errors.mean(axis=1)
else:  # max или отдельный канал
    errors = best_cfg['errors'] if 'errors' in best_cfg and len(best_cfg['errors']) == len(df) else forecast_errors.max(axis=1)

threshold = best_cfg['threshold']

df['forecast_error'] = errors
df['anomaly_score'] = np.clip(errors / (errors.max() + 1e-12), 0, 1)
df['Is_CHRONOS_Anomaly'] = (errors > threshold).astype(int)
df['y_pred'] = df['Is_CHRONOS_Anomaly']

# ====================== МЕТРИКИ (если есть ground truth) ======================
if 'Is_Anomaly' in df.columns:
    df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
    mask_period = df['day_num'] >= ANOMALY_START_DAY
    
    print("\n" + "="*70)
    print("МЕТРИКИ ПРЕДСКАЗАНИЯ")
    print("="*70)
    
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
    
    print(f"\n--- Период аномалий (день {ANOMALY_START_DAY}+) ---")
    y_true_p = df.loc[mask_period, 'y_true']
    y_pred_p = df.loc[mask_period, 'y_pred']
    score_p = df.loc[mask_period, 'anomaly_score']
    
    print(f"F1-score     : {f1_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"Precision    : {precision_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"Recall       : {recall_score(y_true_p, y_pred_p, zero_division=0):.4f}")
    print(f"AUROC        : {roc_auc_score(y_true_p, score_p):.4f}")

# ====================== СТАТИСТИКА ======================
total_anom = df['Is_CHRONOS_Anomaly'].sum()
period_anom = df.loc[df['day_num'] >= ANOMALY_START_DAY, 'Is_CHRONOS_Anomaly'].sum()

print("\n" + "="*65)
print("СТАТИСТИКА ОБНАРУЖЕНИЯ")
print("="*65)
print(f"Всего найдено аномалий: {total_anom} ({total_anom/len(df)*100:.3f}%)")
print(f"Аномалий в периоде {ANOMALY_START_DAY}+ дней: {period_anom}")

# ====================== СОХРАНЕНИЕ ======================
df.to_excel(OUTPUT_FILE, index=False)
print(f"\nПредсказания успешно сохранены в файл:\n{OUTPUT_FILE}")