import numpy as np
import pandas as pd


def get_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic hydrogen genesis classification data based on geological rules.

    Returns a DataFrame with 4 categories x 300 samples each (1200 rows total).
    Columns: delta_D, ln_CH4_H2, R_Ra, C1_C1_plus, delta_13C_CH4, Tectonic_Setting, Genesis_Type
    """
    rng = np.random.default_rng(seed)
    n_per_class = 300

    records = []

    # ── Category 1: 壳源无机 (Crustal Inorganic) ──
    for _ in range(n_per_class):
        records.append({
            "delta_D": rng.uniform(-699, -100),
            "ln_CH4_H2": rng.uniform(-15.0, -8.1),
            "R_Ra": rng.uniform(0.01, 0.09),
            "C1_C1_plus": rng.uniform(0.95, 0.99),
            "delta_13C_CH4": rng.uniform(-30, -10),
            "Tectonic_Setting": "活动断裂带",
            "Genesis_Type": "壳源无机",
        })

    # ── Category 2: 幔源/蛇纹石化 (Mantle / Serpentinization) ──
    for _ in range(n_per_class):
        records.append({
            "delta_D": rng.uniform(-850, -701),
            "ln_CH4_H2": rng.uniform(-7.7, -2.1),
            "R_Ra": rng.uniform(1.1, 8.0),
            "C1_C1_plus": rng.uniform(0.99, 1.0),
            "delta_13C_CH4": rng.uniform(-25, 0),
            "Tectonic_Setting": "超基性岩/洋壳",
            "Genesis_Type": "幔源/蛇纹石化",
        })

    # ── Category 3: 有机质热解 (Thermogenic) ──
    for _ in range(n_per_class):
        records.append({
            "delta_D": rng.uniform(-820, -640),
            "ln_CH4_H2": rng.uniform(-15.0, -10.1),
            "R_Ra": rng.uniform(0.01, 0.1),
            "C1_C1_plus": rng.uniform(0.80, 0.95),
            "delta_13C_CH4": rng.uniform(-50, -35),
            "Tectonic_Setting": "沉积盆地",
            "Genesis_Type": "有机质热解",
        })

    # ── Category 4: 微生物成因 (Microbial) ──
    for _ in range(n_per_class):
        records.append({
            "delta_D": rng.uniform(-830, -700),
            "ln_CH4_H2": rng.uniform(10.0, 15.0),
            "R_Ra": rng.uniform(0.01, 0.05),
            "C1_C1_plus": rng.uniform(0.99, 1.0),
            "delta_13C_CH4": rng.uniform(-80, -55),
            "Tectonic_Setting": "沉积盆地",
            "Genesis_Type": "微生物成因",
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = get_synthetic_data()
    print("Data shape:", df.shape)
    print("\nGenesis_Type distribution:")
    print(df["Genesis_Type"].value_counts())
    print("\nSample rows:")
    print(df.head(10))
