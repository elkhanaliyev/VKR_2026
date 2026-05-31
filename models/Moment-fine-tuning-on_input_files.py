#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOMENT‑1‑small inference

* Читает Excel‑файл, структура которого полностью совпадает с датасетом,
  использованным при обучении (столбцы Time, Is_Anomaly, 4 канала).
* Загружает сохранённые артефакты:
    – StandardScaler (момент обучения)
    – MOMENTPipeline (weights + config)
* Строит скользящие окна длиной SEQ_LEN и получает reconstruction‑error.
* Порог аномалии берётся из первых 60 дн. (перцентиль THRESHOLD_PERCENTILE).
* Метки‑аномалии ставятся **только** для строк, у которых day_num ≥ 61.
* Результат сохраняется в NEW_OUTPUT_FILE с колонками:
      forecast_error, anomaly_score, Is_MOMENT_Anomaly
"""

# -------------------------------------------------
# 0️⃣  IMPORT + SETTINGS
# -------------------------------------------------
import os, warnings, pickle
from datetime import datetime

import numpy as np, pandas as pd, torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, matthews_corrcoef,
                             balanced_accuracy_score, roc_auc_score,
                             average_precision_score, classification_report)

from momentfm import MOMENTPipeline

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())
torch.set_grad_enabled(False)

# ---------- параметры ----------
SEQ_LEN            = 512          # длина окна MOMENT
BATCH_SIZE         = 64
THRESHOLD_PERCENTILE = 99        # перцентиль на train‑периоде
START_DATETIME     = datetime(2025, 10, 5, 0, 2)
ANOMALY_START_DAY  = 61

# ---------- пути ----------
ARTIFACTS_DIR      = "moment_lp_model"                     # где сохранили модель
SCALER_PATH        = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
NEW_INPUT_FILE     = "power_data_correlated_with_anomalies_s1.xlsx"
NEW_OUTPUT_FILE    = "new_power_data_MOMENT_detected.xlsx"

# ---------- каналы ----------
CHANNELS = [
    'Реактивная мощность',
    'Выходной коэффициент мощности',
    'Полная мощность',
    'Ток'
]

# -------------------------------------------------
# 1️⃣  LOAD SCALER & MOMENT MODEL
# -------------------------------------------------
print("\n🔽 Loading scaler …")
with open(SCALER_PATH, "rb") as f:
    scaler: StandardScaler = pickle.load(f)

print("🚀 Loading MOMENT model …")
model = MOMENTPipeline.from_pretrained(
    ARTIFACTS_DIR,                       # <-- папка, где был вызван model.save_pretrained()
    model_kwargs={"task_name": "reconstruction"},
)
model.init()
model.eval()

# -------------------------------------------------
# 2️⃣  READ NEW FILE + PRE‑PROCESS
# -------------------------------------------------
print("\n📂 Reading new file …")
df = pd.read_excel(NEW_INPUT_FILE)

df['Time']    = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1
df[CHANNELS]  = df[CHANNELS].apply(pd.to_numeric, errors='coerce')

# Scale the 4 channels exactly как в training‑pipeline
X_scaled = scaler.transform(df[CHANNELS].values)   # shape (N, 4)

# -------------------------------------------------
# 3️⃣  RECONSTRUCTION‑ERROR (скользящее окно)
# -------------------------------------------------
def reconstruction_errors(X_scaled: np.ndarray,
                          seq_len: int = SEQ_LEN,
                          batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Возвращает массив MAE‑ошибок shape [N, C].
    Окна формируются с шагом = seq_len//2 (можно поменять на 1,
    если нужна более точная оценка).
    """
    N, C = X_scaled.shape
    err_sum   = np.zeros((N, C), dtype=np.float64)
    err_cnt   = np.zeros(N,      dtype=np.float64)

    step = max(1, seq_len // 2)                # шаг между окнами
    starts = list(range(0, N - seq_len + 1, step))
    if not starts:                             # слишком короткая серия
        starts = [0]

    # ---------- батч‑прогон ----------
    with torch.no_grad():
        for i in tqdm(range(0, len(starts), batch_size),
                      desc="MOMENT inference"):
            batch_st = starts[i: i + batch_size]

            # собрать batch → [B, C, seq_len]
            batch_np = np.stack(
                [X_scaled[s: s + seq_len].T for s in batch_st],
                axis=0,                     # (B, C, seq_len)
            )
            x_enc = torch.from_numpy(batch_np).float()          # B×C×L
            mask  = torch.ones(len(batch_st), seq_len, dtype=torch.long)

            out   = model(x_enc=x_enc, input_mask=mask)
            recon = out.reconstruction.cpu().numpy()             # B×C×L

            # MSE per point per channel
            mse = (batch_np - recon) ** 2                        # B×C×L
            mse = mse.transpose(0, 2, 1)                         # B×L×C

            for j, s in enumerate(batch_st):
                err_sum[s: s + seq_len]   += mse[j]              # суммируем
                err_cnt[s: s + seq_len]   += 1

    # ---- покрыть хвост, если он не попал в окно (редко) ----
    tail_start = N - seq_len
    if tail_start not in starts and N >= seq_len:
        win = X_scaled[tail_start: tail_start + seq_len].T[np.newaxis]   # 1×C×L
        x_enc = torch.from_numpy(win).float()
        mask  = torch.ones(1, seq_len, dtype=torch.long)
        out   = model(x_enc=x_enc, input_mask=mask)
        recon = out.reconstruction.cpu().numpy()
        mse = ((win - recon) ** 2).transpose(0, 2, 1)[0]                # L×C
        err_sum[tail_start:]   += mse
        err_cnt[tail_start:]   += 1

    err_cnt = np.maximum(err_cnt, 1)
    return err_sum / err_cnt[:, np.newaxis]          # [N, C]

print("\n🔧 Computing reconstruction errors …")
recon_err = reconstruction_errors(X_scaled)        # shape (N, 4)

# -------------------------------------------------
# 4️⃣  АГРЕГАЦИЯ ОШИБОК (по каналам) 
# -------------------------------------------------
# Вы можете задать свои веса, как в оригинальном коде
CHANNEL_WEIGHTS = np.array([0.4, 0.1, 0.1, 0.3])
agg_error = (recon_err * CHANNEL_WEIGHTS).sum(axis=1)   # единичный вектор [N]

df['forecast_error'] = agg_error

# -------------------------------------------------
# 5️⃣  THRESHOLD (по train‑периоду: day <= 60)
# -------------------------------------------------
train_mask = df['day_num'] <= 60
threshold = np.percentile(df.loc[train_mask, 'forecast_error'].values,
                          THRESHOLD_PERCENTILE)
print(f"\n🔎 Threshold ({THRESHOLD_PERCENTILE}‑pct on train) = {threshold:.6f}")

# -------------------------------------------------
# 6️⃣  METКИ АНОМАЛИИ СТАРТУЮТ С ДНЯ 61
# -------------------------------------------------
df['anomaly_score'] = np.clip(agg_error / (agg_error.max() + 1e-12), 0, 1)

# Бинарный предикт – ставим 1 **только если**:
#   * ошибка превышает порог и
#   * день >= ANOMALY_START_DAY
df['Is_MOMENT_Anomaly'] = 0
mask_anom_start = df['day_num'] >= ANOMALY_START_DAY
df.loc[mask_anom_start, 'Is_MOMENT_Anomaly'] = (
    df.loc[mask_anom_start, 'forecast_error'] > threshold
).astype(int)

# -------------------------------------------------
# 7️⃣  ОТЧЁТ METRIK (если вам нужен, можно закомментировать)
# -------------------------------------------------
if 'Is_Anomaly' in df.columns:                     # есть ground‑truth?
    df['y_true'] = df['Is_Anomaly'].map({'Да': 1, 'Нет': 0}).fillna(0).astype(int)
    y_true = df['y_true'].values
    y_pred = df['Is_MOMENT_Anomaly'].values
    score  = df['anomaly_score'].values

    def _metrics(y_t, y_p, s):
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn + 1e-12)
        return {
            'F1'          : f1_score(y_t, y_p, zero_division=0),
            'Precision'   : precision_score(y_t, y_p, zero_division=0),
            'Recall'      : recall_score(y_t, y_p, zero_division=0),
            'FAR'         : far,
            'MCC'         : matthews_corrcoef(y_t, y_p),
            'BalancedAcc' : balanced_accuracy_score(y_t, y_p),
            'AUROC'       : (roc_auc_score(y_t, s)
                             if len(np.unique(y_t)) > 1 else np.nan),
            'PR_AUC'      : (average_precision_score(y_t, s)
                             if len(np.unique(y_t)) > 1 else np.nan),
            'ConfMat'     : cm,
        }

    all_met = _metrics(y_true, y_pred, score)
    period_met = _metrics(
        y_true[df['day_num'] >= ANOMALY_START_DAY],
        y_pred[df['day_num'] >= ANOMALY_START_DAY],
        score[df['day_num'] >= ANOMALY_START_DAY],
    )

    def _print(m, title):
        print("\n" + "="*70)
        print(f"{title}")
        print("="*70)
        for k in ['F1','Precision','Recall','FAR','MCC','BalancedAcc',
                  'AUROC','PR_AUC']:
            print(f"{k:12s}: {m[k]:.4f}")
        cm = m['ConfMat']
        print("\nConfusion matrix (actual × predicted):")
        print(f"          0      1")
        print(f"   0   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"   1   {cm[1,0]:5d}  {cm[1,1]:5d}")

    _print(all_met,    "METRICS – ВСЕ ДАННЫЕ")
    _print(period_met, f"METRICS – ПЕРИОД С ДНЯ {ANOMALY_START_DAY}")

# -------------------------------------------------
# 8️⃣  SAVE RESULT
# -------------------------------------------------
df.to_excel(NEW_OUTPUT_FILE, index=False)
print(f"\n✅  Done! Result saved → {NEW_OUTPUT_FILE}")