from datetime import datetime
import numpy as np
import pandas as pd

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated_with_anomalies.xlsx"
OUTPUT_FILE = "power_data_correlated_with_anomalies_s3.xlsx"

DRIFT_START_DAY = 61
RAMP_DAYS = 180              # Основная рампа

# Заметный, но контролируемый дрейф
DRIFT_DELTA = np.array([-0.90, +0.050, +23.0, +0.142])# Q, cosφ, S, I
##DRIFT_DELTA = np.array([-0.60, +0.035, +15.0, +0.09])
##DRIFT_DELTA = np.array([-0.90, +0.050, +23.0, +0.142])
##DRIFT_DELTA = np.array([-0.75, +0.0425, +19.0, +0.116])
##DRIFT_DELTA = np.array([-0.82, +0.0485, +21.75, +0.1335])
# Небольшое мультипликативное масштабирование (рост нагрузки)
MULTIPLIER = np.array([1.00, 1.00, 1.00, 1.00])   # S и I растут немного сильнее

COLS = ["Реактивная мощность", "Выходной коэффициент мощности", "Полная мощность", "Ток"]

# ====================== НЕЛИНЕЙНЫЙ ПРОГРЕСС ======================
def get_progress(t, start_time, drift_start_day, ramp_days):
    days = (t - start_time).total_seconds() / 86400.0
    if days < drift_start_day - 1:
        return 0.0
    
    # Нелинейный прогресс: начинается медленно, потом ускоряется
    x = (days - (drift_start_day - 1)) / ramp_days
    # sigmoid-like функция для плавного старта и ускорения
    progress = 1 / (1 + np.exp(-8 * (x - 0.5)))   # центр в середине рампы
    return float(np.clip(progress, 0.0, 1.0))

# ====================== ПРИМЕНЕНИЕ ======================
df = pd.read_excel(INPUT_FILE)
times = pd.to_datetime(df["Time"], format="%d.%m.%Y %H:%M")
start_time = times.iloc[0]
out = df.copy()

print("Применяем заметный, но плавный нелинейный дрейф...")

for i, t in enumerate(times):
    progress = get_progress(t, start_time, DRIFT_START_DAY, RAMP_DAYS)
    
    # Аддитивный дрейф
    delta = progress * DRIFT_DELTA
    
    # Мультипликативный компонент (лёгкий рост)
    mult = 1 + progress * (MULTIPLIER - 1)
    
    q  = out.at[i, COLS[0]] * mult[0] + delta[0]
    c  = out.at[i, COLS[1]] * mult[1] + delta[1]
    s  = out.at[i, COLS[2]] * mult[2] + delta[2]
    it = out.at[i, COLS[3]] * mult[3] + delta[3]
    
    # Ограничения и округление
    out.at[i, COLS[0]] = round(np.clip(q, -11.5, -8.0), 0)
    out.at[i, COLS[1]] = round(np.clip(c, 0.78, 0.98), 3)
    out.at[i, COLS[2]] = int(np.clip(round(s), 35, 280))
    out.at[i, COLS[3]] = round(np.clip(it, 0.18, 1.35), 3)

out.to_excel(OUTPUT_FILE, index=False)

print(f"\nГотово → {OUTPUT_FILE}")
print(f"RAMP_DAYS = {RAMP_DAYS}")

print("\nКорреляции после дрейфа:")
print(out[COLS].corr().round(4))

print("\nСредние значения:")
print("Первые 5000 строк:", out[COLS].iloc[:5000].mean().round(4).to_dict())
print("Последние 5000 строк:", out[COLS].iloc[-5000:].mean().round(4).to_dict())

import matplotlib.pyplot as plt

# Строим progress для каждой точки
progress_values = [get_progress(t, start_time, DRIFT_START_DAY, RAMP_DAYS) 
                   for t in times]

plt.figure(figsize=(14, 4))
plt.plot(times, progress_values, color='steelblue', lw=1.5)
plt.axvline(x=times.iloc[DRIFT_START_DAY * 288], 
            color='red', linestyle='--', label=f'День {DRIFT_START_DAY}')
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='progress=0.5')
plt.title('Функция progress(t) — сигмоидное нарастание дрейфа')
plt.ylabel('progress (0 → 1)')
plt.xlabel('Время')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('progress_curve.png', dpi=150, facecolor='white')
plt.show()
print("Сохранён: progress_curve.png")

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

for i, col in enumerate(COLS):
    # Оригинал и дрейфованный
    orig = pd.Series(df[col].values)
    drift = pd.Series(out[col].values)
    
    window = 288  # одни сутки
    rm_orig  = orig.rolling(window=window, center=True).mean()
    rm_drift = drift.rolling(window=window, center=True).mean()
    
    axes[i].plot(rm_orig,  color='steelblue', lw=1.2, 
                 label='Оригинал', alpha=0.8)
    axes[i].plot(rm_drift, color='darkorange', lw=1.5, 
                 label='После дрейфа')
    axes[i].axvline(x=DRIFT_START_DAY * 288, 
                    color='red', linestyle='--', lw=1, 
                    label=f'День {DRIFT_START_DAY}')
    axes[i].set_ylabel(col, fontsize=9)
    axes[i].grid(alpha=0.3)
    axes[i].legend(fontsize=8, loc='upper left')

axes[-1].set_xlabel('Индекс точки')
plt.suptitle('Rolling mean: оригинал vs дрейфованные данные', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('rolling_mean_drift.png', dpi=150, facecolor='white')
plt.show()

from scipy.stats import wasserstein_distance

# Референс = первые 60 дней
ref_end = DRIFT_START_DAY * 288
ref_data = out[COLS].iloc[:ref_end]

window_size = 288  # одни сутки
step = 288
results = []

for start in range(ref_end, len(out) - window_size, step):
    window = out[COLS].iloc[start:start + window_size]
    t_mid = times.iloc[start + window_size // 2]
    
    wd_per_channel = []
    for col in COLS:
        wd = wasserstein_distance(ref_data[col].dropna(), 
                                  window[col].dropna())
        wd_per_channel.append(wd)
    
    results.append({
        'time': t_mid,
        'day': (t_mid - start_time).days,
        'wd_mean': np.mean(wd_per_channel),
        **{f'wd_{col[:3]}': wd_per_channel[i] 
           for i, col in enumerate(COLS)}
    })

res_df = pd.DataFrame(results)

# График
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(res_df['day'], res_df['wd_mean'], 
        color='darkorange', lw=2, label='Среднее WD по каналам')
for i, col in enumerate(COLS):
    ax.plot(res_df['day'], res_df[f'wd_{col[:3]}'], 
            lw=1, alpha=0.5, linestyle='--', label=col[:15])

ax.set_xlabel('День от начала данных')
ax.set_ylabel('Wasserstein distance')
ax.set_title('Нарастание дрейфа: Wasserstein distance относительно референса')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('wasserstein_drift.png', dpi=150, facecolor='white')
plt.show()

print("\nМонотонность нарастания (первые vs последние 5 окон):")
print("Первые 5 окон WD mean:", res_df['wd_mean'].head(5).round(4).values)
print("Последние 5 окон WD mean:", res_df['wd_mean'].tail(5).round(4).values)
days_total = (times.iloc[-1] - times.iloc[0]).days
print(f"Всего дней в данных: {days_total}")
print(f"Центр сигмоиды: день {DRIFT_START_DAY + RAMP_DAYS // 2}")
print(f"Конец рампы:    день {DRIFT_START_DAY + RAMP_DAYS}")