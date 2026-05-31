#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOMENT-8features Inference + Полные метрики
"""

import os
import warnings
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

from momentfm import MOMENTPipeline

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())
torch.set_grad_enabled(False)

# ====================== НАСТРОЙКИ ======================
SEQ_LEN = 512
BATCH_SIZE = 64
THRESHOLD_PERCENTILE = 99

START_DATETIME = datetime(2025, 10, 5, 0, 2)
ANOMALY_START_DAY = 61

# Пути
ARTIFACTS_DIR = "moment_8features_lp_model"
NEW_INPUT_FILE = "power_data_correlated_with_anomalies_s3.xlsx"   # ← измени при необходимости
NEW_OUTPUT_FILE = "MOMENT_8features_prediction.xlsx"

print("=== MOMENT-8features Inference (с метриками) ===\n")

# ====================== ЗАГРУЗКА МОДЕЛИ ======================
print("Загружаем сохранённую модель...")

with open(os.path.join(ARTIFACTS_DIR, "scaler_level.pkl"), "rb") as f:
    scaler_level = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, "scaler_delta.pkl"), "rb") as f:
    scaler_delta = pickle.load(f)

model = MOMENTPipeline.from_pretrained(
    ARTIFACTS_DIR,
    model_kwargs={"task_name": "reconstruction"}
)
model.init()
model.eval()

print("Модель загружена успешно.\n")

# ====================== ЗАГРУЗКА ДАННЫХ ======================
df = pd.read_excel(NEW_INPUT_FILE)
df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

level_cols = ['Реактивная мощность', 'Выходной коэффициент мощности', 'Полная мощность', 'Ток']
df[level_cols] = df[level_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=level_cols).reset_index(drop=True)
df = df.sort_values('Time').reset_index(drop=True)

# ====================== 8 ПРИЗНАКОВ ======================
df['Q'] = df['Реактивная мощность']
df['cos_phi'] = df['Выходной коэффициент мощности']
df['S'] = df['Полная мощность']
df['I'] = df['Ток']
df['delta_Q'] = df['Q'].diff().fillna(0)
df['delta_cos'] = df['cos_phi'].diff().fillna(0)
df['delta_S'] = df['S'].diff().fillna(0)
df['delta_I'] = df['I'].diff().fillna(0)

# Масштабирование
X_level = df[['Q', 'cos_phi', 'S', 'I']].values
X_delta = df[['delta_Q', 'delta_cos', 'delta_S', 'delta_I']].values

X_scaled = np.hstack([
    scaler_level.transform(X_level),
    scaler_delta.transform(X_delta)
])

# ====================== ИНФЕРЕНС ======================
def get_reconstruction_errors(X_scaled, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    N, C = X_scaled.shape
    err_sum = np.zeros((N, C), dtype=np.float64)
    err_cnt = np.zeros(N, dtype=np.float64)
    step = seq_len // 2
    starts = list(range(0, N - seq_len + 1, step))

    windows = [X_scaled[s:s+seq_len] for s in starts]
    window_starts = starts

    with torch.no_grad():
        for i in tqdm(range(0, len(windows), batch_size), desc="MOMENT Inference"):
            batch_wins = windows[i:i+batch_size]
            batch_starts = window_starts[i:i+batch_size]
            x_np = np.stack(batch_wins, axis=0).transpose(0, 2, 1)
            x_enc = torch.from_numpy(x_np).float()
            mask = torch.ones(len(batch_wins), seq_len, dtype=torch.long)

            out = model(x_enc=x_enc, input_mask=mask)
            recon = out.reconstruction.cpu().numpy()
            mse = ((x_np - recon) ** 2).transpose(0, 2, 1)

            for j, s in enumerate(batch_starts):
                err_sum[s:s+seq_len] += mse[j]
                err_cnt[s:s+seq_len] += 1

    err_cnt = np.maximum(err_cnt, 1)
    return err_sum / err_cnt[:, np.newaxis]

print("\nВычисляем ошибки реконструкции...")
recon_errors = get_reconstruction_errors(X_scaled)

# ====================== АГРЕГАЦИЯ ======================
weights = np.array([0.25, 0.1, 0.25, 0.15, 0.1, 0.05, 0.05, 0.05])
df['recon_error'] = (recon_errors * weights).sum(axis=1)

# Порог
train_mask = df['day_num'] <= 60
threshold = np.percentile(df.loc[train_mask, 'recon_error'].values, THRESHOLD_PERCENTILE)
print(f"Порог аномалии ({THRESHOLD_PERCENTILE}-й перцентиль): {threshold:.6f}")

df['anomaly_score'] = np.clip(df['recon_error'] / (df['recon_error'].max() + 1e-12), 0, 1)

# Метки аномалий только с 61 дня
df['Is_MOMENT_Anomaly'] = 0
anom_mask = df['day_num'] >= ANOMALY_START_DAY
df.loc[anom_mask, 'Is_MOMENT_Anomaly'] = (df.loc[anom_mask, 'recon_error'] > threshold).astype(int)

# ====================== МЕТРИКИ ======================
df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
df['y_pred'] = df['Is_MOMENT_Anomaly']

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
    print("\n" + "="*75)
    if set_name:
        print(f"РЕЗУЛЬТАТЫ: {set_name}")
    print("="*75)
    print(f"F1-score     : {metrics['f1']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f}")
    print(f"Recall       : {metrics['recall']:.4f}")
    print(f"FAR          : {metrics['far']:.4f}")
    print(f"MCC          : {metrics['mcc']:.4f}")
    print(f"Balanced Acc : {metrics['balanced_acc']:.4f}")
    print(f"AUROC        : {metrics['auroc']:.4f}")
    print(f"PR-AUC       : {metrics['prauc']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"Normal   {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"Anomaly  {cm[1,0]:6d} {cm[1,1]:6d}")
    print("="*75)

# ====================== ВЫВОД МЕТРИК ======================
mask_period = df['day_num'] >= ANOMALY_START_DAY

metrics_all = compute_all_metrics(df['y_true'], df['y_pred'], df['anomaly_score'])
print_metrics(metrics_all, "MOMENT-8features — ВСЕ ДАННЫЕ")

metrics_period = compute_all_metrics(
    df.loc[mask_period, 'y_true'],
    df.loc[mask_period, 'y_pred'],
    df.loc[mask_period, 'anomaly_score']
)
print_metrics(metrics_period, f"MOMENT-8features — ПЕРИОД АНОМАЛИЙ (день {ANOMALY_START_DAY}+)")

# ====================== СОХРАНЕНИЕ ======================
df.to_excel(NEW_OUTPUT_FILE, index=False)

print(f"\n✅ Результат успешно сохранён в: {NEW_OUTPUT_FILE}")
print(f"Найдено аномалий всего: {df['Is_MOMENT_Anomaly'].sum()} "
      f"({df['Is_MOMENT_Anomaly'].mean()*100:.2f}%)")
print(f"Из них в периоде аномалий: {df.loc[mask_period, 'Is_MOMENT_Anomaly'].sum()}")