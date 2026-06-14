"""
从原始文献 Excel 文件中提取真实地球化学数据，与合成数据集合并。

读取 6 个关键 Excel 文件，提取或计算 5 个数值特征 + 构造背景，
用文献规则标注成因类型，缺失参数以类中位数填充。
"""
import os
import sys
import re
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── 路径 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = r"C:\Users\。\Desktop\大创氢气\数据"
SYNTHETIC_CSV = os.path.join(BASE_DIR, "hydrogen_training_dataset.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "hydrogen_combined_dataset.csv")

# ── 辅助函数 ───────────────────────────────────────────
def safe_float(val):
    """将各种格式转换为浮点数，失败返回 NaN。"""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if val in ("", "-", "一", "–", "—", "痕量", "–", "<0.01", "n.d."):
            return np.nan
        # 归一化 Unicode 减号 (U+2212) 为 ASCII 减号
        val = val.replace("−", "-")
        # 去除 < > 约 等前缀
        val = re.sub(r"^[<>≈约≤≥～~]\s*", "", val)
        # 范围取中间值（支持 ASCII 减号和 Unicode 减号 U+2212）
        m = re.match(r"([\-−]?\d+\.?\d*)\s*[～~\-—―–]\s*([\-−]?\d+\.?\d*)", val)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2
        try:
            return float(val)
        except ValueError:
            return np.nan
    return np.nan


def calc_ln_ch4_h2(ch4_pct, h2_pct):
    """计算 ln(CH₄/H₂)。CH₄ 或 H₂ 为 0 时返回 NaN。"""
    if pd.isna(ch4_pct) or pd.isna(h2_pct):
        return np.nan
    if ch4_pct <= 0 or h2_pct <= 0:
        return np.nan
    return np.log(ch4_pct / h2_pct)


def assign_origin(row):
    """基于文献规则自动分配成因类型。"""
    dD = row.get("delta_D", np.nan)
    ln_ratio = row.get("ln_CH4_H2", np.nan)
    ra = row.get("R_Ra", np.nan)
    tectonic = row.get("Tectonic_Setting", "")

    # R/Ra 主导判断
    if not pd.isna(ra) and ra >= 0.1:
        return "幔源/蛇纹石化"

    # δD 主导判断
    if not pd.isna(dD):
        if dD > -700:
            # 壳源无机方向
            if not pd.isna(ln_ratio) and ln_ratio >= 10:
                return "微生物成因"
            return "壳源无机"
        else:
            # δD < -700: 幔源 或 生物成因 或 热解
            if not pd.isna(ln_ratio):
                if ln_ratio < -10:
                    return "有机质热解"
                elif ln_ratio >= 10:
                    return "微生物成因"
            # 构造背景辅助
            if "超基性" in str(tectonic) or "洋壳" in str(tectonic):
                return "幔源/蛇纹石化"
            if "沉积盆地" in str(tectonic):
                return "有机质热解"
            return "幔源/蛇纹石化"

    # ln(CH₄/H₂) 主导判断（无 δD 时）
    if not pd.isna(ln_ratio):
        if ln_ratio < -10:
            return "有机质热解"
        elif ln_ratio >= 10:
            return "微生物成因"
        elif ln_ratio > -4:
            return "幔源/蛇纹石化"
        else:
            return "壳源无机"

    # 构造背景兜底
    if "超基性" in str(tectonic) or "洋壳" in str(tectonic) or "蛇绿" in str(tectonic):
        return "幔源/蛇纹石化"
    if "沉积盆地" in str(tectonic):
        return "有机质热解"
    return "壳源无机"


def assign_tectonic(geological_context):
    """根据地质背景映射到构造背景类别。"""
    ctx = str(geological_context)
    if any(w in ctx for w in ["蛇绿", "超基性", "洋壳", "蛇纹", "橄榄岩", "玄武", "辉长岩"]):
        if any(w in ctx for w in ["Kansas", "Scott", "Heins"]):
            return "超基性岩/洋壳"  # Kansas 是蛇纹石化成因
        return "超基性岩/洋壳"
    if any(w in ctx for w in ["盆地", "坳陷", "凹陷", "沙河子", "登娄库", "营城",
                               "侏罗", "古近", "柴达木", "松辽", "鄂尔多斯", "四川",
                               "渤海湾", "沁水", "三水", "楚雄", "黔", "Amadeus"]):
        return "沉积盆地"
    if any(w in ctx for w in ["断裂", "火山", "热海", "腾冲", "长白山", "天池",
                               "基岩", "基底", "花岗岩", "变质"]):
        return "活动断裂带"
    return "沉积盆地"


# ═══════════════════════════════════════════════════════════════
# 各文件提取函数
# ═══════════════════════════════════════════════════════════════

def extract_qaidam(base_dir):
    """柴达木盆地阿尔金山前带 — 参数最全。"""
    path = os.path.join(base_dir, "柴达木盆地阿尔金山前带天然气中组分及含量.xlsx")
    df = pd.read_excel(path, header=None)

    records = []
    # 数据从 Row 4 开始，Row 2-3 是表头
    for i in range(4, len(df)):
        row = df.iloc[i]
        # 跳过空行
        if pd.isna(row.iloc[0]) and pd.isna(row.iloc[4]):
            continue

        seq = row.iloc[0]        # 序号
        field = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ""
        well = str(row.iloc[2]) if not pd.isna(row.iloc[2]) else ""
        layer = str(row.iloc[4]) if not pd.isna(row.iloc[4]) else ""

        h2_pct = safe_float(row.iloc[5])
        dD = safe_float(row.iloc[6])

        # 组分列：CH₄(7), C₂(8), C₃(9), CO₂(10), N₂(11), He(12)
        ch4_pct = safe_float(row.iloc[7])
        c2 = safe_float(row.iloc[8])
        c3 = safe_float(row.iloc[9])
        co2 = safe_float(row.iloc[10])
        n2 = safe_float(row.iloc[11])
        he = safe_float(row.iloc[12])
        delta_13C = safe_float(row.iloc[15])
        # δ²H₁(CH₄) 在 col 16，不需要

        # 跳过无关键数据的行
        if pd.isna(dD) and pd.isna(h2_pct) and pd.isna(ch4_pct):
            continue

        ln_ratio = calc_ln_ch4_h2(ch4_pct, h2_pct)
        c1_c1plus = ch4_pct / (ch4_pct + c2 + c3) if (not pd.isna(ch4_pct) and not pd.isna(c2)) else np.nan

        tectonic = assign_tectonic(f"{field} {layer} 柴达木盆地")

        records.append({
            "delta_D": dD if not pd.isna(dD) else np.nan,
            "ln_CH4_H2": ln_ratio if not pd.isna(ln_ratio) else np.nan,
            "R_Ra": np.nan,  # 此表无 R/Ra
            "C1_C1_plus": c1_c1plus if not pd.isna(c1_c1plus) else np.nan,
            "delta_13C_CH4": delta_13C if not pd.isna(delta_13C) else np.nan,
            "Tectonic_Setting": tectonic,
            "Source": f"柴达木-{field}-{well}",
        })

    return records


def extract_songliao(base_dir):
    """松辽盆地松科2井 — 有 H₂% 和 CH₄%，可算 ln(CH₄/H₂)。"""
    path = os.path.join(base_dir, "松辽盆地松科2井天然气气体组分.xlsx")
    df = pd.read_excel(path, header=None)

    records = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        depth = safe_float(row.iloc[0])
        formation = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ""
        lithology = str(row.iloc[2]) if not pd.isna(row.iloc[2]) else ""
        ch4_pct = safe_float(row.iloc[3])
        c2plus = safe_float(row.iloc[4])
        co2_pct = safe_float(row.iloc[5])
        n2_pct = safe_float(row.iloc[6])
        he_pct = safe_float(row.iloc[7])
        h2_pct = safe_float(row.iloc[8])

        if pd.isna(h2_pct) or pd.isna(ch4_pct):
            continue

        ln_ratio = calc_ln_ch4_h2(ch4_pct, h2_pct)
        c1_c1plus = ch4_pct / (ch4_pct + c2plus) if (not pd.isna(ch4_pct) and not pd.isna(c2plus) and c2plus > 0) else np.nan

        records.append({
            "delta_D": np.nan,
            "ln_CH4_H2": ln_ratio,
            "R_Ra": np.nan,
            "C1_C1_plus": c1_c1plus,
            "delta_13C_CH4": np.nan,
            "Tectonic_Setting": "沉积盆地",
            "Source": f"松辽-SK2-{formation}-{int(depth)}m" if not pd.isna(depth) else f"松辽-SK2-{formation}",
        })

    return records


def extract_ophiolite(base_dir):
    """蛇绿岩杂岩渗漏 — 有 H₂ Mol% 和 CH₄ Mol%。"""
    path = os.path.join(base_dir, "蛇绿岩杂岩中减少的天然气渗漏：H₂-CH₄-N₂混合气多成因的证据.xlsx")
    df = pd.read_excel(path, header=None)

    records = []
    current_type = ""
    current_country = ""

    for i in range(1, len(df)):
        row = df.iloc[i]
        # 检查是否是类型分隔行
        col0 = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ""

        if "type" in col0:
            current_type = col0
            continue

        # 检查是否是国家名行（后面跟着有效数据）
        site = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ""
        h2_val = safe_float(row.iloc[8])
        ch4_val = safe_float(row.iloc[10])
        n2_val = safe_float(row.iloc[9])
        co2_val = safe_float(row.iloc[11])

        # 判断当前国家
        if col0 in ("Oman", "Philippines", "New Caledonia", "Turkey"):
            current_country = col0
            # 有些行国家名和 site 名在同一行
            if pd.isna(h2_val) and pd.isna(ch4_val):
                continue

        if pd.isna(h2_val) or h2_val <= 0:
            continue

        ln_ratio = calc_ln_ch4_h2(ch4_val, h2_val)

        records.append({
            "delta_D": np.nan,
            "ln_CH4_H2": ln_ratio,
            "R_Ra": np.nan,
            "C1_C1_plus": np.nan,
            "delta_13C_CH4": np.nan,
            "Tectonic_Setting": "超基性岩/洋壳",
            "Source": f"蛇绿岩-{current_country}-{site}",
        })

    return records


def extract_kansas(base_dir):
    """Kansas盆地 Scott/Heins 井 — 蛇纹石化作用相关。"""
    path = os.path.join(base_dir, "北美Kansas盆地Scott和Heins井氢气藏井中气体组分分析——蛇纹石化作用.xlsx")
    df = pd.read_excel(path, header=None)

    records = []
    current_well = ""
    sample_idx = 0

    for i in range(2, len(df)):
        row = df.iloc[i]
        col0 = row.iloc[0]

        if isinstance(col0, str) and col0 in ("Scott", "Heins"):
            current_well = col0
            sample_idx = 0
            # 井名行：col1=日期, col2=He, col3=H₂, col4=O₂, col5=N₂, col6=CO₂, col7=CH₄, col8=Ar
            h2_val = safe_float(row.iloc[3])
            ch4_val = safe_float(row.iloc[7])
        else:
            # 后续行：col0=日期, col1=He, col2=H₂, col3=O₂, col4=N₂, col5=CO₂, col6=CH₄, col7=Ar
            h2_val = safe_float(row.iloc[2])
            ch4_val = safe_float(row.iloc[6])

        if pd.isna(h2_val) or h2_val <= 0:
            continue

        sample_idx += 1
        ln_ratio = calc_ln_ch4_h2(ch4_val, h2_val)

        records.append({
            "delta_D": np.nan,
            "ln_CH4_H2": ln_ratio,
            "R_Ra": np.nan,
            "C1_C1_plus": np.nan,
            "delta_13C_CH4": np.nan,
            "Tectonic_Setting": "超基性岩/洋壳",
            "Source": f"Kansas-{current_well}-{sample_idx}",
        })

    return records


def extract_china_h2(base_dir):
    """中国天然氢显示统计 — 有 δD 范围数据。"""
    path = os.path.join(base_dir, "中国天然氢显示统计.xlsx")
    df = pd.read_excel(path, header=None)

    records = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        basin = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ""
        area = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ""
        h2_str = str(row.iloc[2]) if not pd.isna(row.iloc[2]) else ""
        dD_str = str(row.iloc[3]) if not pd.isna(row.iloc[3]) else ""

        # 提取 δD 中值
        dD = safe_float(dD_str)
        h2_val = safe_float(h2_str)

        if pd.isna(dD) and pd.isna(h2_val):
            continue

        tectonic = assign_tectonic(f"{basin} {area}")

        records.append({
            "delta_D": dD if not pd.isna(dD) else np.nan,
            "ln_CH4_H2": np.nan,
            "R_Ra": np.nan,
            "C1_C1_plus": np.nan,
            "delta_13C_CH4": np.nan,
            "Tectonic_Setting": tectonic,
            "Source": f"中国-{basin}-{area}",
        })

    return records


def extract_amadeus(base_dir):
    """Amadeus 盆地 — 有 R/Ra 和气体组分。"""
    path = os.path.join(base_dir, "Amadeus 盆地.xlsx")
    df = pd.read_excel(path, header=None)

    records = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        well = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ""
        depth = safe_float(row.iloc[1])
        h2_val = safe_float(row.iloc[6])
        ch4_val = safe_float(row.iloc[9])
        ra_val = safe_float(row.iloc[14])

        if pd.isna(h2_val) or h2_val <= 0:
            continue

        ln_ratio = calc_ln_ch4_h2(ch4_val, h2_val)

        records.append({
            "delta_D": np.nan,
            "ln_CH4_H2": ln_ratio,
            "R_Ra": ra_val if not pd.isna(ra_val) else np.nan,
            "C1_C1_plus": np.nan,
            "delta_13C_CH4": np.nan,
            "Tectonic_Setting": "沉积盆地",
            "Source": f"Amadeus-{well}-{int(depth)}m" if not pd.isna(depth) else f"Amadeus-{well}",
        })

    return records


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  原始数据提取 & 数据集合并")
    print("=" * 60)

    # 1. 提取各文件
    print("\n[1/4] 从 6 个 Excel 文件提取数据...")
    extractors = [
        ("柴达木盆地阿尔金山前带", extract_qaidam),
        ("松辽盆地松科2井", extract_songliao),
        ("蛇绿岩杂岩渗漏", extract_ophiolite),
        ("Kansas盆地Scott/Heins井", extract_kansas),
        ("中国天然氢显示统计", extract_china_h2),
        ("Amadeus盆地", extract_amadeus),
    ]

    all_records = []
    for name, func in extractors:
        try:
            records = func(DATA_DIR)
            print(f"  {name}: {len(records)} 条")
            all_records.extend(records)
        except Exception as e:
            print(f"  {name}: 提取失败 - {e}")

    df_real = pd.DataFrame(all_records)
    print(f"\n  总计提取: {len(df_real)} 条真实数据")

    # 2. 用文献规则自动标注 Origin
    print("\n[2/4] 文献规则自动标注成因类型...")
    df_real["Origin"] = df_real.apply(assign_origin, axis=1)

    # 3. 缺失值用合成数据类中位数填充
    print("[3/4] 缺失参数用合成数据类中位数填充...")
    df_syn = pd.read_csv(SYNTHETIC_CSV)
    num_cols = ["delta_D", "ln_CH4_H2", "R_Ra", "C1_C1_plus", "delta_13C_CH4"]
    class_medians = df_syn.groupby("Origin")[num_cols].median()

    for col in num_cols:
        missing_count = df_real[col].isna().sum()
        if missing_count > 0:
            for origin in df_real["Origin"].unique():
                if origin not in class_medians.index:
                    continue
                median_val = class_medians.loc[origin, col]
                mask = (df_real["Origin"] == origin) & (df_real[col].isna())
                df_real.loc[mask, col] = median_val
            print(f"  {col}: 填充了 {missing_count} 个缺失值")

    # 4. 合并
    print("[4/4] 合并合成数据与真实数据...")
    syn_cols = num_cols + ["Tectonic_Setting", "Origin"]
    df_combined = pd.concat(
        [df_syn[syn_cols], df_real[syn_cols]],
        ignore_index=True,
    )
    # 打乱顺序
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # 统计
    print(f"\n  合成数据: {len(df_syn)} 条")
    print(f"  真实数据: {len(df_real)} 条")
    print(f"  合并总计: {len(df_combined)} 条")

    print(f"\n  合并后类别分布:")
    for label, count in df_combined["Origin"].value_counts().items():
        print(f"    {label}: {count}")

    print(f"\n  真实数据类别分布:")
    for label, count in df_real["Origin"].value_counts().items():
        print(f"    {label}: {count}")

    # 保存
    df_combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n  合并数据集已保存至: {OUTPUT_CSV}")
    print(f"  文件大小: {os.path.getsize(OUTPUT_CSV) / 1024:.1f} KB")

    # 保存真实数据单独文件（方便查看）
    real_csv = os.path.join(BASE_DIR, "hydrogen_real_data.csv")
    df_real.to_csv(real_csv, index=False, encoding="utf-8")
    print(f"  真实数据已保存至: {real_csv}")

    print("\n" + "=" * 60)
    print("  完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
