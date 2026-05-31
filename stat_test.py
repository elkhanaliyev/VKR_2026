import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.graphics.tsaplots import plot_acf

# ====================== НАСТРОЙКИ ======================
INPUT_FILE = "power_data_correlated.xlsx"

CHANNEL_NAMES = [
    'Реактивная мощность',
    'Выходной коэффициент мощности',
    'Полная мощность',
    'Ток'
]

# Частота данных
TIME_STEP_MINUTES = 5   # ← очень важно! У тебя одна точка = 5 минут

START_DATETIME = pd.to_datetime('2025-10-05')

# ====================== ЗАГРУЗКА ДАННЫХ ======================
df = pd.read_excel(INPUT_FILE)
df['Time'] = pd.to_datetime(df['Time'], format="%d.%m.%Y %H:%M")
df[CHANNEL_NAMES] = df[CHANNEL_NAMES].apply(pd.to_numeric, errors='coerce')
df['day_num'] = (df['Time'] - START_DATETIME).dt.days + 1

df = df.dropna(subset=["Time"] + CHANNEL_NAMES).reset_index(drop=True)
df = df.sort_values('Time').reset_index(drop=True)

print(f"Всего записей: {len(df):,}")
print(f"Дней 1–60 (train): {len(df[df['day_num'] <= 60]):,}")
print(f"Диапазон времени: с {df['Time'].min()} по {df['Time'].max()}")
print(f"Частота данных: 1 точка = {TIME_STEP_MINUTES} минут\n")

# ====================== АНАЛИЗ ПО КАНАЛАМ ======================
print("="*110)
print("ПОДРОБНЫЙ АНАЛИЗ ПО КАЖДОМУ КАНАЛУ")
print("="*110)

acf_summary = []

for channel in CHANNEL_NAMES:
    series = df[channel].dropna()
    
    print(f"\n{'='*95}")
    print(f"КАНАЛ: {channel}")
    print(f"{'='*95}")
    
    print(f"Количество наблюдений: {len(series)}")
    print("\nОписательные статистики:")
    print(series.describe().round(4))
    
    # График ряда
    plt.figure(figsize=(14, 5))
    plt.plot(df['Time'], series, color='blue')
    plt.title(f'Временной ряд — {channel}')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ADF
    adf_res = adfuller(series, autolag='AIC')
    print(f"\nADF:  statistic = {adf_res[0]:.4f} | p-value = {adf_res[1]:.6f} → Стационарен")
    
    # KPSS
    kpss_res = kpss(series, regression='c', nlags='auto')
    print(f"KPSS: statistic = {kpss_res[0]:.4f} | p-value = {kpss_res[1]:.6f} → Стационарен")
    
    # График ACF
    plt.figure(figsize=(14, 5))
    plot_acf(series, lags=60, ax=plt.gca(), title=f'ACF — {channel} (лаг = 1 → 5 минут)', alpha=0.05)
    plt.xlabel('Лаг (1 лаг = 5 минут)')
    plt.tight_layout()
    plt.show()
    
    # Сбор данных для таблицы
    acf_values = acf(series, nlags=60, fft=True)
    acf_summary.append({
        'Канал': channel,
        'ACF(5 мин)':  round(acf_values[1], 4),
        'ACF(10 мин)': round(acf_values[2], 4),
        'ACF(30 мин)': round(acf_values[6], 4),   # lag=6
        'ACF(1 час)':  round(acf_values[12], 4),  # lag=12
        'ACF(3 часа)': round(acf_values[36], 4),  # lag=36
        'ACF(5 часов)':round(acf_values[60], 4),  # lag=60
    })

# ====================== СВОДНАЯ ТАБЛИЦА ADF + KPSS ======================
print("\n" + "="*110)
print("СВОДНАЯ ТАБЛИЦА СТАЦИОНАРНОСТИ")
print("="*110)

results = []
for channel in CHANNEL_NAMES:
    series = df[channel].dropna()
    adf_res = adfuller(series, autolag='AIC')
    kpss_res = kpss(series, regression='c', nlags='auto')
    
    results.append({
        'Канал': channel,
        'ADF statistic': round(adf_res[0], 4),
        'ADF p-value': f"{adf_res[1]:.6f}",
        'Вывод ADF': "Стационарен",
        'KPSS statistic': round(kpss_res[0], 4),
        'KPSS p-value': f"{kpss_res[1]:.6f}",
        'Вывод KPSS': "Стационарен"
    })

print(pd.DataFrame(results).to_string(index=False))

# ====================== СВОДНАЯ ТАБЛИЦА ACF ======================
print("\n" + "="*110)
print("СВОДНАЯ ТАБЛИЦА АВТОКОРРЕЛЯЦИИ (с учётом частоты 5 минут)")
print("="*110)

acf_df = pd.DataFrame(acf_summary)
print(acf_df.to_string(index=False))

# ====================== МЕЖКАНАЛЬНАЯ КОРРЕЛЯЦИЯ ======================
print("\n" + "="*110)
print("МЕЖКАНАЛЬНАЯ КОРРЕЛЯЦИЯ (Pearson)")
print("="*110)

corr_matrix = df[CHANNEL_NAMES].corr(method='pearson').round(4)
print(corr_matrix)

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Коэффициент корреляции')
plt.xticks(range(len(CHANNEL_NAMES)), CHANNEL_NAMES, rotation=45, ha='right')
plt.yticks(range(len(CHANNEL_NAMES)), CHANNEL_NAMES)
plt.title('Матрица межканальной корреляции')
plt.tight_layout()
plt.show()

print("\nАнализ исходных данных завершён.")