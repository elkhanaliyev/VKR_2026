from datetime import datetime
import pandas as pd
import numpy as np
import os
import pickle
import torch
from chronos import BaseChronosPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())
torch.set_grad_enabled(False)

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated_with_anomalies_s3.xlsx"      # ← измените на ваш новый файл
OUTPUT_FILE = "power_data_CHRONOS_4features_predictions.xlsx"
SCALER_PATH = "chronos_scaler.pkl"
BEST_CFG_PATH = "chronos_best_cfg.pkl"

ANOMALY_START_DAY = 61
CONTEXT_LEN = 864
PRED_LEN = 12
BATCH_SIZE = 32
NUM_SAMPLES = 20

START_DATETIME = datetime(2025, 10, 5, 0, 2)

print("=== Chronos-Bolt (4 features) — Inference Script ===\n")

# ====================== ЗАГРУЗКА МОДЕЛИ И АРТЕФАКТОВ ======================
print("Загружаем скейлер и лучшую конфигурацию...")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(BEST_CFG_PATH, "rb") as f:
    best_cfg = pickle.load(f)

print("Артефакты загружены успешно.")

# ====================== ЗАГРУЗКА CHRONOS ======================
print("\nЗагружаем Chronos-Bolt-tiny...")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",
    device_map="cpu",
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
)
print("Chronos-Bolt успешно загружен.\n")

# ====================== ЗАГРУЗКА НОВОГО ФАЙЛА ======================
df = pd.read_excel(INPUT_FILE)
df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

CHANNEL_NAMES = ['Реактивная мощность', 'Выходной коэффициент мощности',
                 'Полная мощность', 'Ток']

df[CHANNEL_NAMES] = df[CHANNEL_NAMES].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=CHANNEL_NAMES).reset_index(drop=True)
df = df.sort_values('Time').reset_index(drop=True)

print(f"Загружено записей: {len(df):,}")

# ====================== МАСШТАБИРОВАНИЕ ======================
X_full = df[CHANNEL_NAMES].values
X_full_scaled = scaler.transform(X_full)

# ====================== ИНФЕРЕНС ======================
def get_forecast_errors(X_scaled, context_len=CONTEXT_LEN, pred_len=PRED_LEN, num_samples=NUM_SAMPLES):
    N, C = X_scaled.shape
    error_sum = np.zeros((N, C), dtype=np.float64)
    error_count = np.zeros(N, dtype=np.float64)
    step = pred_len
    starts = list(range(context_len, N - pred_len + 1, step))

    for ch in range(C):
        ch_series = X_scaled[:, ch]
        for i in tqdm(range(0, len(starts), BATCH_SIZE), desc=f"Chronos [ch_{ch}]"):
            batch_starts = starts[i:i + BATCH_SIZE]
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

    # Заполняем начало
    train_mean_err = errors[:len(df[df['day_num'] <= 60])].mean(axis=0)
    errors[:context_len] = train_mean_err

    return errors


print("\nВычисляем ошибки прогнозирования на новых данных...")
forecast_errors = get_forecast_errors(X_full_scaled)

# ====================== ПРИМЕНЕНИЕ ЛУЧШЕЙ КОНФИГУРАЦИИ ======================
best_errors = best_cfg['errors']
threshold = best_cfg['threshold']

df['forecast_error'] = best_errors
df['anomaly_score'] = np.clip(best_errors / (best_errors.max() + 1e-12), 0, 1)
df['Is_CHRONOS_Anomaly'] = (best_errors > threshold).astype(int)

# ====================== МЕТКИ ======================
df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
df['y_pred'] = df['Is_CHRONOS_Anomaly']

# ====================== МЕТРИКИ ======================
if 'Is_Anomaly' in df.columns:
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

total_anom = df['Is_CHRONOS_Anomaly'].sum()
period_anom = df.loc[df['day_num'] >= ANOMALY_START_DAY, 'Is_CHRONOS_Anomaly'].sum()

print(f"Всего найдено аномалий: {total_anom} ({total_anom/len(df)*100:.3f}%)")
print(f"Аномалий в периоде {ANOMALY_START_DAY}+ дней: {period_anom}")

if 'y_true' in df.columns:
    true_period = df.loc[df['day_num'] >= ANOMALY_START_DAY, 'y_true'].sum()
    print(f"Истинных аномалий в периоде: {true_period}")

# ====================== СОХРАНЕНИЕ ======================
df.to_excel(OUTPUT_FILE, index=False)
print(f"\n✅ Предсказания успешно сохранены в файл:\n{OUTPUT_FILE}")