#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chronos‑2 inference on a new power‑data file.
Steps:
  1) load scaler + best‑cfg (saved after the grid‑search stage)
  2) read a new Excel file (same columns as training data)
  3) rebuild the forecast‑error series exactly as in the training pipeline
  4) apply the saved aggregation / weights / threshold
  5) print standard classification metrics (F1, Precision, Recall,
     FAR, MCC, Balanced‑Accuracy, AUROC, PR‑AUC) and the confusion matrix
  6) save the file with additional columns:
        forecast_error, anomaly_score, Is_CHRONOS2_Anomaly
"""

# -------------------------------------------------
# 0️⃣  IMPORT + GLOBAL SETTINGS
# -------------------------------------------------
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, matthews_corrcoef,
                             balanced_accuracy_score, roc_auc_score,
                             average_precision_score)

from chronos import Chronos2Pipeline   # <-- Chronos‑2

warnings.filterwarnings("ignore")
torch.set_num_threads(min(8, os.cpu_count()))   # ограничиваем количество потоков

# ------------------- параметры окна -------------------
CONTEXT_LEN = 144          # 12 ч при 1‑мин шаге
PRED_LEN    = 12           # предсказываем 12 минут вперёд
BATCH_SIZE  = 128          # можно увеличить (см. комментарий ниже)

# ------------------- имена каналов --------------------
CHANNEL_NAMES = [
    'Реактивная мощность',
    'Выходной коэффициент мощности',
    'Полная мощность',
    'Ток'
]

# ------------------- дата‑отсчёта --------------------
START_DATETIME = datetime(2025, 10, 5, 0, 2)   # должна совпадать с training‑script

# ------------------- пути к артефактам ---------------
SCALER_PATH = "chronos2_scaler.pkl"      # <‑‑ файл, который был сохранён после обучения
CFG_PATH    = "chronos2_best_cfg.pkl"    # <‑‑ конфигурация (aggregation, weights, …)

# ------------------- файл для инференса -------------
NEW_INPUT_FILE  = "power_data_correlated_with_anomalies_s2.xlsx"   # <-- замените на ваш файл
NEW_OUTPUT_FILE = "new_power_data_CHRONOS2_detected.xlsx"

# -------------------------------------------------
# 1️⃣  LOAD SAVED ARTIFACTS
# -------------------------------------------------
with open(SCALER_PATH, "rb") as f:
    scaler: StandardScaler = pickle.load(f)

with open(CFG_PATH, "rb") as f:
    best_cfg: dict = pickle.load(f)      # keys: aggregation, weights, percentile, threshold

print("\n=== Loaded artefacts ===")
print(f"Scaler  : trained on {scaler.mean_.shape[0]} features")
print(f"Config  : aggregation={best_cfg['aggregation']}, "
      f"weights={best_cfg.get('weights')}, "
      f"percentile={best_cfg['percentile']}, "
      f"threshold={best_cfg['threshold']:.4f}")

# -------------------------------------------------
# 2️⃣  READ & PRE‑PROCESS NEW DATASET
# -------------------------------------------------
print("\n=== Reading new file ===")
df = pd.read_excel(NEW_INPUT_FILE)

# Приводим время к datetime и считаем номер дня (должно совпадать с training‑script)
df['Time']    = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

# Приводим каналы к числовому типу (float)
df[CHANNEL_NAMES] = df[CHANNEL_NAMES].apply(pd.to_numeric, errors='coerce')

# Скалируем те же признаки, что использовались при обучении
X_scaled = scaler.transform(df[CHANNEL_NAMES].values)   # shape (N, C)

# -------------------------------------------------
# 3️⃣  HELPERS: format detection & median extraction
# -------------------------------------------------
def detect_format(df: pd.DataFrame) -> str:
    """Определяет, в каком виде Chronos‑2 возвращает predict_df."""
    cols = set(df.columns)
    if 'quantile_level' in cols:
        return 'long'
    if 0.5 in cols or '0.5' in cols:
        return 'wide'
    if 'mean' in cols or 'median' in cols:
        return 'mean'
    # fallback – ищем числовые имена колонок‑квантили
    numeric = [c for c in df.columns
               if isinstance(c, (float, int)) or
               (isinstance(c, str) and c.replace('.','',1).isdigit())]
    return 'wide' if numeric else 'unknown'


def extract_median(pred_df: pd.DataFrame, ts_id: str,
                   pred_len: int, fmt: str) -> np.ndarray:
    """Возвращает массив медианных предсказаний (квантиль 0.5)."""
    sub = pred_df[pred_df["id"] == ts_id]
    if fmt == 'long':
        qcol = 'quantile_level'
        valcol = 'target' if 'target' in sub.columns else sub.columns[-1]
        med = sub[sub[qcol] == 0.5][valcol].values
        if len(med) == 0:
            med = sub[sub[qcol] == '0.5'][valcol].values
        if len(med) == 0:
            # берём ближайший к 0.5
            closest = min(sub[qcol].unique(),
                          key=lambda x: abs(float(x) - 0.5))
            med = sub[sub[qcol] == closest][valcol].values
        return med[:pred_len].astype(np.float32)

    if fmt == 'wide':
        key = 0.5 if 0.5 in sub.columns else '0.5'
        return sub[key].values[:pred_len].astype(np.float32)

    if fmt == 'mean':
        key = 'median' if 'median' in sub.columns else 'mean'
        return sub[key].values[:pred_len].astype(np.float32)

    # fallback – возьмём последнюю числовую колонку
    num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ('quantile_level', 'index')]
    return sub[num_cols[-1]].values[:pred_len].astype(np.float32)


# -------------------------------------------------
# 4️⃣  FORECAST‑ERROR CALCULATION (как в training‑script)
# -------------------------------------------------
def get_forecast_errors(X_scaled: np.ndarray,
                        context_len: int = CONTEXT_LEN,
                        pred_len: int    = PRED_LEN,
                        batch_size: int  = BATCH_SIZE) -> np.ndarray:
    """
    Возвращает массив ошибок MAE shape=(N, C).
    Алгоритм полностью копирует то, что использовалось в обучении:
      * скользящее окно контекста,
      * предсказываем `pred_len` точек,
      * берём медианный прогноз,
      * усредняем MAE по всем покрывающим окнам.
    """
    N, C = X_scaled.shape
    err_sum   = np.zeros((N, C), dtype=np.float64)
    err_count = np.zeros(N,      dtype=np.float64)

    # -------------------------------------------------
    # 1️⃣  Определяем формат вывода модели (одноразово)
    # -------------------------------------------------
    dummy = np.sin(np.linspace(0, 4 * np.pi, context_len))
    ts_dummy = pd.date_range("2000-01-01", periods=context_len, freq="1min")
    tmp_df = pd.DataFrame({"id": "tmp_0", "timestamp": ts_dummy, "target": dummy})
    fmt_check = pipeline.predict_df(
        tmp_df,
        prediction_length=pred_len,
        quantile_levels=[0.5],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )
    pred_format = detect_format(fmt_check)

    # -------------------------------------------------
    # 2️⃣  Список стартов окон
    # -------------------------------------------------
    step = pred_len                       # каждое предсказание начинается каждые pred_len точек
    starts = list(range(context_len, N - pred_len + 1, step))

    print(f"\n🔍 predict_df format → '{pred_format}'")
    print(f"🔄 Processing {len(starts)} windows (batch‑size={batch_size}) …")

    # -------------------------------------------------
    # 3️⃣  Основной цикл по батчам
    # -------------------------------------------------
    for i in tqdm(range(0, len(starts), batch_size), desc="Batch inference"):
        batch_starts = starts[i: i + batch_size]

        # ----- 3.1 Формируем «long» DataFrame (одна строка = один timestamp) -----
        rows = []
        for b_idx, s in enumerate(batch_starts):
            for ch in range(C):
                window = X_scaled[s - context_len: s, ch]          # shape (context_len,)
                ts = pd.date_range("2000-01-01", periods=context_len, freq="1min")
                # Каждый элемент окна – отдельная строка
                for t, val in zip(ts, window.astype(np.float32)):
                    rows.append({
                        "id"       : f"ts_{b_idx}_{ch}",
                        "timestamp": t,
                        "target"   : float(val),          # скаляр, тип float
                    })
        batch_df = pd.DataFrame(rows)

        # ----- 3.2 Прогноз модели -----
        pred_df = pipeline.predict_df(
            batch_df,
            prediction_length=pred_len,
            quantile_levels=[0.5],          # нам нужна только медиана
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # ----- 3.3 Вычисляем MAE для всех (id, ch) в батче -----
        for b_idx, s in enumerate(batch_starts):
            for ch in range(C):
                ts_id = f"ts_{b_idx}_{ch}"
                actual = X_scaled[s: s + pred_len, ch]
                median_forecast = extract_median(pred_df, ts_id,
                                                 pred_len, pred_format)
                if median_forecast.size == 0:
                    continue          # защита от пустого прогноза
                n = min(len(actual), len(median_forecast))
                mae = np.abs(actual[:n] - median_forecast[:n])
                err_sum[s: s + n, ch]   += mae
                err_count[s: s + n]     += 1

    # -------------------------------------------------
    # 4️⃣  Финальная нормализация
    # -------------------------------------------------
    err_count = np.maximum(err_count, 1)          # защита от деления на 0
    errors = err_sum / err_count[:, np.newaxis]   # shape (N, C)

    # Первые context_len точек не покрыты прогнозом – заполняем их средней ошибкой
    # по тренировочному периоду (дни 1‑60), точно так же, как делали в training‑script
    train_mask = df['day_num'] <= 60
    train_mean_err = errors[train_mask.values].mean(axis=0)
    errors[:context_len] = train_mean_err

    return errors


# -------------------------------------------------
# 5️⃣  LOAD Chronos‑2 model
# -------------------------------------------------
print("\n🚀 Loading Chronos‑2 …")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",          # замените на "auto" если есть GPU
    dtype=torch.float32,
)

# -------------------------------------------------
# 6️⃣  CALCULATE FORECAST ERRORS
# -------------------------------------------------
print("\n=== Computing forecast errors ===")
forecast_errors = get_forecast_errors(X_scaled)   # shape (N, C)

# -------------------------------------------------
# 7️⃣  AGGREGATE ERRORS according to saved best_cfg
# -------------------------------------------------
agg_type = best_cfg['aggregation']
weights  = best_cfg.get('weights')    # может быть None

if agg_type == 'max':
    agg_errors = forecast_errors.max(axis=1)          # [N]
elif agg_type == 'mean':
    agg_errors = forecast_errors.mean(axis=1)         # [N]
elif agg_type == 'weighted' and isinstance(weights, np.ndarray):
    agg_errors = (forecast_errors * weights).sum(axis=1)   # [N]
else:                                   # фиксированный канал, например 'ch_2'
    # ожидаем строку вида 'ch_0', 'ch_1', …
    try:
        ch_idx = int(agg_type.split('_')[-1])
        agg_errors = forecast_errors[:, ch_idx]
    except Exception:
        # fallback – берём среднее, если формат не распознан
        agg_errors = forecast_errors.mean(axis=1)

# -------------------------------------------------
# 8️⃣  BUILD ANOMALY SCORE & LABEL
# -------------------------------------------------
threshold = best_cfg['threshold']
df['forecast_error'] = agg_errors
df['anomaly_score']  = np.clip(agg_errors / (agg_errors.max() + 1e-12), 0, 1)
df['Is_CHRONOS2_Anomaly'] = (agg_errors > threshold).astype(int)

# -------------------------------------------------
# 9️⃣  METRICS (весь набор + только период аномалий)
# -------------------------------------------------
def compute_metrics(y_true, y_pred, score):
    """Возвращает словарь со всем набором метрик."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn + 1e-12)

    metrics = {
        'F1'           : f1_score(y_true, y_pred, zero_division=0),
        'Precision'    : precision_score(y_true, y_pred, zero_division=0),
        'Recall'       : recall_score(y_true, y_pred, zero_division=0),
        'FAR'          : far,
        'MCC'          : matthews_corrcoef(y_true, y_pred),
        'BalancedAcc'  : balanced_accuracy_score(y_true, y_pred),
        'AUROC'        : (roc_auc_score(y_true, score)
                         if len(np.unique(y_true)) > 1 else np.nan),
        'PR_AUC'       : (average_precision_score(y_true, score)
                         if len(np.unique(y_true)) > 1 else np.nan),
        'ConfMatrix'   : cm
    }
    return metrics


# Истинные метки (если они есть в исходном файле)
if 'Is_Anomaly' in df.columns:
    df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
else:
    # если меток нет – создаём столбец‑заглушку, чтобы ниже не падало
    df['y_true'] = 0

# Метрики **по всем строкам**
metrics_all = compute_metrics(
    y_true = df['y_true'].values,
    y_pred = df['Is_CHRONOS2_Anomaly'].values,
    score  = df['anomaly_score'].values
)

# Метрики **только для периода, где начинаются аномалии** (day >= 61)
mask_anom_period = df['day_num'] >= 61
metrics_anom = compute_metrics(
    y_true = df.loc[mask_anom_period, 'y_true'].values,
    y_pred = df.loc[mask_anom_period, 'Is_CHRONOS2_Anomaly'].values,
    score  = df.loc[mask_anom_period, 'anomaly_score'].values
)

# -------------------------------------------------
# 10️⃣  PRINT REPORT
# -------------------------------------------------
def print_report(met, title: str):
    print("\n" + "=" * 70)
    print(f"   {title}")
    print("=" * 70)
    print(f"F1‑score                : {met['F1']:.4f}")
    print(f"Precision               : {met['Precision']:.4f}")
    print(f"Recall                  : {met['Recall']:.4f}")
    print(f"FAR (False Alarm Rate)  : {met['FAR']:.4f}")
    print(f"MCC                     : {met['MCC']:.4f}")
    print(f"Balanced Accuracy       : {met['BalancedAcc']:.4f}")
    print(f"AUROC (anomaly_score)   : {met['AUROC']:.4f}")
    print(f"PR‑AUC (anomaly_score)  : {met['PR_AUC']:.4f}")

    cm = met['ConfMatrix']
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print("               Pred 0   Pred 1")
    print(f"Actual 0   {cm[0,0]:7d}   {cm[0,1]:7d}")
    print(f"Actual 1   {cm[1,0]:7d}   {cm[1,1]:7d}")
    print("=" * 70)


print_report(metrics_all,  "METRICS – ВСЕ ДАННЫЕ")
print_report(metrics_anom, f"METRICS – ПЕРИОД АНОМАЛИЙ (day >= {61})")

# -------------------------------------------------
# 11️⃣  SAVE RESULT
# -------------------------------------------------
df.to_excel(NEW_OUTPUT_FILE, index=False)
print(f"\n✅  Result saved → {NEW_OUTPUT_FILE}")

# -------------------------------------------------
# 12️⃣  OPTIONAL: пересчёт порога на новых данных
# -------------------------------------------------
# Если вы хотите адаптировать порог под текущий набор,
# раскомментируйте блок ниже (установите `if True`).
# -----------------------------------------------------------------
# if True:
#     train_err = agg_errors[df['day_num'] <= 60]
#     new_thr   = np.percentile(train_err, best_cfg['percentile'])
#     print(f"\n[INFO] Re‑computed threshold: {new_thr:.4f} (old {threshold:.4f})")
#     df['Is_CHRONOS2_Anomaly'] = (agg_errors > new_thr).astype(int)
#     # пересчёт метрик с новым порогом
#     metrics_all = compute_metrics(df['y_true'].values,
#                                   df['Is_CHRONOS2_Anomaly'].values,
#                                   df['anomaly_score'].values)
#     print_report(metrics_all, "METRICS – С НОВЫМ ПОРОГОМ")
# -----------------------------------------------------------------

# -------------------------------------------------
# 13️⃣  OPTIONAL: quick visual check
# -------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(df["Time"], agg_errors, label="Aggregated MAE")
    plt.axhline(threshold, color="red", linestyle="--",
                label=f"Threshold ({best_cfg['percentile']}‑pct)")
    plt.title("Chronos‑2 | Аномалии (day ≥ 61)")
    plt.xlabel("Time")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.show()