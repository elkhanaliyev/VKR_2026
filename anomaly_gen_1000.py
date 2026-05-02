import numpy as np
import pandas as pd

# ============ НАСТРОЙКИ power_data_correlated_with_anomalies======================
INPUT_FILE = "power_data_correlated.xlsx" ##"power_data_fitted_correlation.xlsx"
OUTPUT_FILE = "power_data_correlated_with_anomalies.xlsx"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MIN_DAY_FOR_ANOMALY = 61
MINUTES_PER_DAY = 24 * 12

# ====================== ЗАГРУЗКА ======================
df = pd.read_excel(INPUT_FILE)

# ====================== КОЛОНКИ ======================
df['Is_Anomaly'] = 'Нет'
df['anomaly_type'] = 'normal'

# ====================== КОВАРИАЦИЯ ======================
anomaly_cov = np.array([
    [0.2,   -0.001,  5.0,   0.1],
    [-0.001, 0.002,  0.5,   0.01],
    [5.0,    0.5,   50.0,  2.0],
    [0.1,    0.01,   2.0,  0.05]
])

channels = [
    'Реактивная мощность',
    'Выходной коэффициент мощности',
    'Полная мощность',
    'Ток'
]

# ====================== БЛОКИ ======================
block_A = [
    [-9, 0.856, 46, 0.195, 'Нет', 'normal'],
    [-13.2, 0.769, 196, 0.629, 'Да', 'block'],
    [-9, 0.861, 46, 0.203, 'Да', 'block']
]

block_B = [
    [-9, 0.856, 46, 0.195, 'Нет', 'normal'],
    [-11, 0.988, 366, 1.601, 'Да', 'block'],
    [-10, 0.96, 133, 0.582, 'Да', 'block'],
    [-9, 0.859, 46, 0.204, 'Нет', 'normal']
]

# ====================== НОВАЯ ФУНКЦИЯ ======================
def insert_block(df, start_idx, block):
    for i, row in enumerate(block):
        idx = start_idx + i
        if idx >= len(df):
            break

        base = np.array(row[:4])

        # Разный уровень шума
        if row[4] == 'Да':
            noise_scale = 0.05   # аномалии
        else:
            noise_scale = 0.02   # нормальные

        noise = np.random.multivariate_normal(
            mean=np.zeros(4),
            cov=anomaly_cov * noise_scale
        )

        sample = base + noise

        df.loc[idx, channels[0]] = round(sample[0], 1)
        df.loc[idx, channels[1]] = round(sample[1], 3)
        df.loc[idx, channels[2]] = int(sample[2])
        df.loc[idx, channels[3]] = round(sample[3], 3)

        df.loc[idx, 'Is_Anomaly'] = row[4]
        df.loc[idx, 'anomaly_type'] = row[5]

# ====================== ИНДЕКСЫ ======================
min_idx_for_anomaly = MIN_DAY_FOR_ANOMALY * MINUTES_PER_DAY

available_indices = np.arange(min_idx_for_anomaly, len(df)-5)
np.random.shuffle(available_indices)

# ====================== БЛОКИ A ======================
inserted_count = 0
i = 0
while inserted_count < 250 and i < len(available_indices):
    idx = available_indices[i]

    overlap = False
    for j in range(idx, idx+len(block_A)):
        if j < len(df) and df.loc[j, 'Is_Anomaly'] == 'Да':
            overlap = True
            break

    if not overlap:
        insert_block(df, idx, block_A)
        inserted_count += 1

    i += 1

# ====================== БЛОКИ B ======================
inserted_count = 0
i = 0
while inserted_count < 250 and i < len(available_indices):
    idx = available_indices[i]

    overlap = False
    for j in range(idx, idx+len(block_B)):
        if j < len(df) and df.loc[j, 'Is_Anomaly'] == 'Да':
            overlap = True
            break

    if not overlap:
        insert_block(df, idx, block_B)
        inserted_count += 1

    i += 1

# ====================== СОХРАНЕНИЕ ======================
df.to_excel(OUTPUT_FILE, index=False)
print(f"Файл с 100 аномалиями сохранён: {OUTPUT_FILE}")