import numpy as np
import pandas as pd
from scipy import stats
from datetime import timedelta

# ========================= НАСТРОЙКИ =========================
SCENARIOS = {
    "raw_data": "power_data_correlated.xlsx",
    "s0":       "power_data_correlated_with_anomalies.xlsx",
    "s1":       "power_data_correlated_with_anomalies_s1.xlsx",
    "s2":       "power_data_correlated_with_anomalies_s2.xlsx",
    "s3":       "power_data_correlated_with_anomalies_s3.xlsx",
}

time_col = "Time"
feature_cols = [
    "Реактивная мощность",
    "Выходной коэффициент мощности",
    "Полная мощность",
    "Ток",
]

GOLDEN_DAYS = 60
WINDOW_DAYS = 5
STEP_DAYS = 5

# ========================= ФУНКЦИИ =========================
def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    
    for col in feature_cols:
        df[col] = (df[col].astype(str)
                   .str.replace(",", ".", regex=False)
                   .str.replace(" ", "", regex=False)
                   .str.replace(r"[^\d\.-]", "", regex=True))
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=[time_col] + feature_cols).sort_values(time_col).reset_index(drop=True)
    return df


def get_window_means(df):
    """Возвращает список средних по каждому скользящему окну (после золотого периода)"""
    golden_end = df[time_col].iloc[0] + timedelta(days=GOLDEN_DAYS)
    dt = df[time_col].diff().median()
    pts_win = int(pd.Timedelta(days=WINDOW_DAYS) / dt)
    pts_step = int(pd.Timedelta(days=STEP_DAYS) / dt)
    
    window_means = {col: [] for col in feature_cols}
    
    idx = 0
    while idx + pts_win <= len(df):
        seg = df.iloc[idx:idx + pts_win]
        if seg[time_col].iloc[0] >= golden_end:   # только после опорного периода
            for col in feature_cols:
                window_means[col].append(seg[col].mean())
        idx += pts_step
    
    return {col: np.array(vals) for col, vals in window_means.items()}


def welch_test(ref_means, test_means, channel):
    """Welch's t-test: тест на то, что среднее в тестовом периоде > опорного"""
    t_stat, p_value = stats.ttest_ind(test_means, ref_means, equal_var=False, alternative='greater')
    
    mean_ref = ref_means.mean()
    mean_test = test_means.mean()
    rel_shift = (mean_test - mean_ref) / abs(mean_ref) * 100 if mean_ref != 0 else np.nan
    
    return {
        "channel": channel,
        "mean_opor": round(mean_ref, 4),
        "mean_test": round(mean_test, 4),
        "rel_shift_%": round(rel_shift, 2),
        "t_stat": round(t_stat, 4),
        "p_value": p_value,
        "significance": "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    }


# ========================= ОСНОВНОЙ ЗАПУСК =========================
print("=== Welch's t-тест: сравнение средних по окнам относительно опорного периода ===\n")

golden_df = load_data(SCENARIOS["raw_data"])  # опорный период берём из чистых данных
golden_means = get_window_means(golden_df)   # на самом деле нам нужны средние только из опорного, но функция возвращает после — исправим ниже

# Правильно: средние опорного периода считаем один раз
golden_end = golden_df[time_col].iloc[0] + timedelta(days=GOLDEN_DAYS)
golden_data = golden_df[golden_df[time_col] < golden_end]

ref_means_dict = {col: golden_data[col].values for col in feature_cols}  # весь опорный период

results = []

for scen_name, path in SCENARIOS.items():
    print(f"Обрабатывается: {scen_name} ...")
    df = load_data(path)
    window_means = get_window_means(df)   # средние по 24 окнам после 60-го дня
    
    for col in feature_cols:
        test_means = window_means[col]
        ref_means = ref_means_dict[col]   # сравниваем с полным опорным периодом
        
        res = welch_test(ref_means, test_means, col)
        res["scenario"] = scen_name
        results.append(res)

# ========================= ВЫВОД ТАБЛИЦЫ =========================
df_results = pd.DataFrame(results)

# Красивая сводная таблица
pivot = df_results.pivot(index="channel", columns="scenario", values=["rel_shift_%", "t_stat", "p_value", "significance"])

print("\n" + "="*100)
print("Таблица 11 — Результаты Welch's t-теста (alternative = greater)")
print("="*100)
print(pivot.round(4).to_string())

# Сохранение
df_results.to_excel("t_test_rel_shift.xlsx", index=False)
print("\nРезультаты сохранены в t_test_rel_shift.xlsx")

# Короткая сводка для текста магистерской
print("\nКраткая сводка для текста:")
for scen in SCENARIOS.keys():
    print(f"\n{scen.upper()}:")
    sub = df_results[df_results["scenario"] == scen]
    for _, row in sub.iterrows():
        print(f"  {row['channel']:30} rel_shift = {row['rel_shift_%']:6.2f}%   p = {row['p_value']:.2e}  {row['significance']}")