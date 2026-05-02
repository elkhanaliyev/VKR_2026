from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib.pyplot as plt

# ========================= НАСТРОЙКИ =========================
INPUT_FILE = "power_data_correlated_with_anomalies_s2.xlsx"
SHEET_NAME = 0
time_col = "Time"
feature_cols = [
    "Реактивная мощность",
    "Выходной коэффициент мощности",
    "Полная мощность",
    "Ток",
]

WINDOW_DAYS = 5
STEP_DAYS = 5
GOLDEN_DAYS = 60                      # первые 60 дней — золотой стандарт

# Пороги
REL_SHIFT_THRESHOLDS = {
    "Реактивная мощность": 0.03,
    "Выходной коэффициент мощности": 0.01,
    "Полная мощность": 0.05,
    "Ток": 0.05,
}

WD_THRESHOLDS = {
    "Реактивная мощность": 0.10,
    "Выходной коэффициент мощности": 0.003,
    "Полная мощность": 1.0,
    "Ток": 0.005,
}

KS_P_THRESHOLD = 0.001
CORR_DIFF_THRESHOLD = 0.12
MIN_FEATURE_FLAGS_FOR_WINDOW = 2
CONSECUTIVE_WINDOWS_FOR_DRIFT = 2     # уменьшил до 2 — более разумно
COOLDOWN_WINDOWS = 2

# ========================= ЗАГРУЗКА =========================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, engine="openpyxl")

df.columns = [c.strip() for c in df.columns]
df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")

for col in feature_cols:
    df[col] = (df[col].astype(str)
               .str.replace(",", ".", regex=False)
               .str.replace(" ", "", regex=False)
               .str.replace(r"[^\d\.-]", "", regex=True))
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[time_col] + feature_cols).sort_values(time_col).reset_index(drop=True)
df[time_col] = df[time_col].dt.round("min")
df = df.drop_duplicates(subset=[time_col]).reset_index(drop=True)

print(f"Всего загружено строк: {len(df)}")
print(f"Период: {df[time_col].iloc[0]} — {df[time_col].iloc[-1]}\n")

# ========================= ЗОЛОТОЙ СТАНДАРТ =========================
golden_end_date = df[time_col].iloc[0] + timedelta(days=GOLDEN_DAYS)
golden_df = df[df[time_col] < golden_end_date].copy()

print(f"Золотой стандарт ({GOLDEN_DAYS} дней): {len(golden_df)} точек")
print(f"Золотой период заканчивается: {golden_end_date.date()}\n")

if len(golden_df) < 500:
    raise ValueError("Слишком мало данных в золотом стандарте!")

golden_data = golden_df[feature_cols]

# ========================= СОЗДАНИЕ ОКОН (только после золотого стандарта) =========================
dt = df[time_col].diff().median()
points_per_window = int(pd.Timedelta(days=WINDOW_DAYS) / dt)
step_points = int(pd.Timedelta(days=STEP_DAYS) / dt)

windows = []
idx = 0
while idx + points_per_window <= len(df):
    seg = df.iloc[idx:idx + points_per_window]
    
    # Берем только окна, которые полностью после золотого стандарта
    if seg[time_col].iloc[0] >= golden_end_date:
        windows.append({
            "window_id": len(windows),
            "t_start": seg[time_col].iloc[0],
            "t_mid": seg[time_col].iloc[len(seg)//2],
            "t_end": seg[time_col].iloc[-1],
            "data": seg[feature_cols].copy()
        })
    
    idx += step_points

print(f"Создано окон для анализа после золотого стандарта: {len(windows)}\n")

if len(windows) == 0:
    raise ValueError("Не создано ни одного окна после золотого стандарта! Увеличьте общий период данных или уменьшите GOLDEN_DAYS.")

# ========================= МЕТРИКИ =========================
def feature_metrics(ref: pd.Series, cur: pd.Series):
    ref = ref.dropna()
    cur = cur.dropna()
    if len(ref) < 30 or len(cur) < 30:
        return {"ks_p": np.nan, "wd": np.nan, "rel_shift": np.nan}
    _, ks_p = ks_2samp(ref, cur)
    wd = wasserstein_distance(ref, cur)
    rel_shift = (cur.mean() - ref.mean()) / (abs(ref.mean()) + 1e-12)
    return {"ks_p": ks_p, "wd": wd, "rel_shift": rel_shift}

rows = []
for win in windows:
    cur = win["data"]
    row = {
        "window_id": win["window_id"],
        "t_start": win["t_start"],
        "t_mid": win["t_mid"],
        "t_end": win["t_end"],
    }
    
    feature_flags = []
    for col in feature_cols:
        m = feature_metrics(golden_data[col], cur[col])
        
        row[f"{col}_ks_p"] = m["ks_p"]
        row[f"{col}_wd"] = m["wd"]
        row[f"{col}_rel_shift"] = m["rel_shift"]
        
        rel_bad = (not np.isnan(m["rel_shift"])) and (abs(m["rel_shift"]) >= REL_SHIFT_THRESHOLDS.get(col, 0.05))
        wd_bad = (not np.isnan(m["wd"])) and (m["wd"] >= WD_THRESHOLDS.get(col, 1.0))
        ks_bad = (not np.isnan(m["ks_p"])) and (m["ks_p"] < KS_P_THRESHOLD)
        
        flag = int(rel_bad or wd_bad)
        
        row[f"{col}_rel_bad"] = int(rel_bad)
        row[f"{col}_wd_bad"] = int(wd_bad)
        row[f"{col}_ks_bad"] = int(ks_bad)
        row[f"{col}_flag"] = flag
        feature_flags.append(flag)
    
    # Структурный дрейф
    golden_corr = golden_data.corr()
    cur_corr = cur.corr()
    corr_abs_diff = (cur_corr - golden_corr).abs()
    mask = ~np.eye(len(feature_cols), dtype=bool)
    corr_diff = corr_abs_diff.values[mask].mean()
    
    row["corr_diff"] = corr_diff
    row["struct_flag"] = int(corr_diff >= CORR_DIFF_THRESHOLD)
    
    # Итоговые флаги
    row["feature_flags_count"] = sum(feature_flags)
    row["any_feature_flag"] = int(row["feature_flags_count"] >= MIN_FEATURE_FLAGS_FOR_WINDOW)
    row["combined_flag_raw"] = int(row["any_feature_flag"] or row["struct_flag"])
    
    rows.append(row)

metrics_df = pd.DataFrame(rows)

# ========================= ПОДТВЕРЖДЕНИЕ =========================
if len(metrics_df) > 0:
    flags = metrics_df["combined_flag_raw"].fillna(0).astype(int).values
    run_len = np.zeros(len(flags), dtype=int)
    confirmed = np.zeros(len(flags), dtype=int)
    cooldown_left = 0
    current_run = 0

    for i in range(len(flags)):
        if cooldown_left > 0:
            cooldown_left -= 1
            run_len[i] = 0
            continue
        
        if flags[i] == 1:
            current_run += 1
        else:
            current_run = 0
        
        run_len[i] = current_run
        
        if current_run >= CONSECUTIVE_WINDOWS_FOR_DRIFT:
            confirmed[i] = 1
            cooldown_left = COOLDOWN_WINDOWS
            current_run = 0

    metrics_df["consecutive"] = run_len
    metrics_df["drift_confirmed"] = confirmed

    # Номера событий
    event_id = 0
    event_ids = [event_id := event_id + 1 if v == 1 else 0 for v in confirmed]
    metrics_df["drift_event_id"] = event_ids

# ========================= ВЫВОД =========================
print("=" * 90)
n_raw = int(metrics_df["combined_flag_raw"].sum()) if not metrics_df.empty else 0
n_conf = int(metrics_df["drift_confirmed"].sum()) if not metrics_df.empty else 0

print(f"Анализ относительно золотого стандарта ({GOLDEN_DAYS} дней)")
print(f"Окон проанализировано: {len(metrics_df)}")
print(f"Raw сигналов: {n_raw} | Подтверждённых дрейфов: {n_conf}")

if n_conf > 0:
    first_time = metrics_df.loc[metrics_df["drift_confirmed"] == 1, "t_mid"].iloc[0]
    print(f"✅ Первый дрейф обнаружен: {first_time}")
else:
    print("❌ Дрейф относительно золотого стандарта не подтверждён.")

# Сохранение
metrics_df.to_csv("drift_vs_golden.tsv", sep="\t", index=False, encoding="utf-8")
print("\nРезультат сохранён в: drift_vs_golden.tsv")