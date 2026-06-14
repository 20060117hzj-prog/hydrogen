"""
标准数据集构建脚本 — 基于文献参数边界生成含真实重叠的合成数据。

v2.0 核心改进（对比 v1.0 干净版本）：
  1. 每类成因按概率分配多种构造背景（不再唯一映射）
  2. 数值参数范围向相邻类别扩展，制造重叠区
  3. 模拟文献中已知的模糊边界（如幔源/蛇纹石化 vs 微生物成因在 δD<-700‰ 重叠）

输出: hydrogen_realistic_dataset.csv (4 类 × 500 条 = 2000 条)
"""
import numpy as np
import pandas as pd
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "hydrogen_realistic_dataset.csv")
SEED = 42
N_PER_CLASS = 500

# ═══════════════════════════════════════════════════════════════
# 类别定义（含构造背景概率分布 + 扩展后的参数范围）
# ═══════════════════════════════════════════════════════════════
CLASS_DEFS = [
    (
        "壳源无机",
        {"活动断裂带": 0.65, "沉积盆地": 0.25, "超基性岩/洋壳": 0.10},
        {
            "delta_D":       (-720, -100),
            "ln_CH4_H2":     (-15.0, -7.5),
            "R_Ra":          (0.01, 0.15),
            "C1_C1_plus":    (0.92, 0.99),
            "delta_13C_CH4": (-32, 0),
        },
    ),
    (
        "幔源/蛇纹石化",
        {"超基性岩/洋壳": 0.65, "活动断裂带": 0.20, "沉积盆地": 0.15},
        {
            "delta_D":       (-850, -650),
            "ln_CH4_H2":     (-8.0, -2.0),
            "R_Ra":          (0.5, 8.0),
            "C1_C1_plus":    (0.95, 1.0),
            "delta_13C_CH4": (-35, 5),
        },
    ),
    (
        "有机质热解",
        {"沉积盆地": 0.65, "活动断裂带": 0.20, "超基性岩/洋壳": 0.15},
        {
            "delta_D":       (-840, -600),
            "ln_CH4_H2":     (-15.0, -7.5),
            "R_Ra":          (0.01, 0.20),
            "C1_C1_plus":    (0.78, 0.96),
            "delta_13C_CH4": (-55, -28),
        },
    ),
    (
        "微生物成因",
        {"沉积盆地": 0.60, "活动断裂带": 0.20, "超基性岩/洋壳": 0.20},
        {
            "delta_D":       (-850, -650),
            "ln_CH4_H2":     (5.0, 15.0),
            "R_Ra":          (0.01, 0.12),
            "C1_C1_plus":    (0.94, 1.0),
            "delta_13C_CH4": (-82, -50),
        },
    ),
]


def build_dataset() -> pd.DataFrame:
    """生成 4×500=2000 条含参数重叠的合成数据。"""
    rng = np.random.default_rng(SEED)
    records = []

    for origin, tectonic_probs, ranges in CLASS_DEFS:
        tectonic_labels = list(tectonic_probs.keys())
        tectonic_weights = list(tectonic_probs.values())

        for _ in range(N_PER_CLASS):
            tectonic = rng.choice(tectonic_labels, p=tectonic_weights)
            records.append({
                "delta_D":         rng.uniform(*ranges["delta_D"]),
                "ln_CH4_H2":       rng.uniform(*ranges["ln_CH4_H2"]),
                "R_Ra":            rng.uniform(*ranges["R_Ra"]),
                "C1_C1_plus":      rng.uniform(*ranges["C1_C1_plus"]),
                "delta_13C_CH4":   rng.uniform(*ranges["delta_13C_CH4"]),
                "Tectonic_Setting": tectonic,
                "Origin":           origin,
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("  真实化数据集构建 — hydrogen_realistic_dataset.csv")
    print("=" * 60)

    df = build_dataset()

    print(f"\n  总样本数: {len(df)}")
    print(f"  特征列:   {list(df.columns[:-1])}")
    print(f"  标签列:   Origin")

    print(f"\n  类别分布:")
    for label, count in df["Origin"].value_counts().items():
        print(f"    {label}: {count}")

    num_cols = ["delta_D", "ln_CH4_H2", "R_Ra", "C1_C1_plus", "delta_13C_CH4"]
    print(f"\n  数值特征统计（全局）:")
    print(df[num_cols].describe().to_string())

    # 构造背景 × 成因交叉表
    print(f"\n  构造背景 × 成因 交叉表:")
    cross = pd.crosstab(df["Tectonic_Setting"], df["Origin"])
    print(cross.to_string())

    # 关键重叠区
    print(f"\n  参数重叠验证:")
    mantle_dD = df[df["Origin"] == "幔源/蛇纹石化"]["delta_D"]
    microbial_dD = df[df["Origin"] == "微生物成因"]["delta_D"]
    lo, hi = max(mantle_dD.min(), microbial_dD.min()), min(mantle_dD.max(), microbial_dD.max())
    if lo < hi:
        n = len(df[df["Origin"].isin(["幔源/蛇纹石化", "微生物成因"]) & df["delta_D"].between(lo, hi)])
        print(f"    δD 幔源/微生物重叠区 [{lo:.0f}, {hi:.0f}]: {n} 条")
    for origin in df["Origin"].unique():
        tectonics = list(df[df["Origin"] == origin]["Tectonic_Setting"].unique())
        print(f"    {origin} 构造背景: {tectonics}")

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"\n  数据集已保存至: {CSV_PATH}")
    print(f"  文件大小: {os.path.getsize(CSV_PATH) / 1024:.1f} KB")
    print("\n" + "=" * 60)
    print("  完成！可运行 train_model.py 训练模型。")
    print("=" * 60)


if __name__ == "__main__":
    main()
