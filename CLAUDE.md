# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

**氢气成因多维智能判识系统** (Hydrogen Genesis Multi-Dimensional Intelligent Identifier) — a desktop application that uses a Random Forest classifier to identify the origin of natural hydrogen gas based on geochemical parameters. This is part of a 大学生创新训练计划 (College Student Innovation Training Program) project at 长江大学, running 2025.11–2026.11.

## Pipeline

```
extract_real_data.py  ──→  hydrogen_real_data.csv
                                │
build_dataset.py  ──→  hydrogen_realistic_dataset.csv  ──┬──→  hydrogen_combined_dataset.csv
                                                         │
                                                         ├──→  train_model.py  ──→  hydrogen_model.pkl  ──→  main_app.py
                                                         │
                                                         └──→  compare_models.py
```

1. **`build_dataset.py`** — 基于文献参数范围生成 2000 条合成训练数据（4 类 × 500 条），v2.0 包含参数重叠和多构造背景混合，输出 `hydrogen_realistic_dataset.csv`。
2. **`extract_real_data.py`** — 从 `C:\Users\。\Desktop\大创氢气\数据\` 下的 6 个 Excel 文件提取真实样品数据（约 102 条），用文献规则自动标注成因，缺失参数以类中位数填充，输出 `hydrogen_real_data.csv` 和合并后的 `hydrogen_combined_dataset.csv`（2102 条）。
3. **`train_model.py`** — 读取 CSV，独热编码，训练随机森林（150 棵树），默认用真实化数据集，支持命令行指定数据集路径。
4. **`main_app.py`** — 桌面 GUI（CustomTkinter，暗色主题），3 个标签页。
5. **`compare_models.py`** — 9 种分类器横向对比（5 折交叉验证），支持命令行指定数据集路径。

> **注意:** `data_generator.py` 是早期版本（每类 300 条，列名 `Genesis_Type`），已被 `build_dataset.py` 取代。

## Running

```bash
# 1. 构建训练数据集
python build_dataset.py

# 2. 提取真实文献数据 & 合并数据集
python extract_real_data.py

# 3. 训练模型（可选数据集）
python train_model.py                          # 默认真实化数据集
python train_model.py hydrogen_combined_dataset.csv  # 合并数据集

# 4. (可选) 模型对比评估
python compare_models.py                              # 默认真实化数据集
python compare_models.py hydrogen_combined_dataset.csv  # 合并数据集

# 5. 启动桌面应用
python main_app.py
```

## 真实数据概况

[extract_real_data.py](extract_real_data.py) 从 6 个文献 Excel 文件中提取了 **102 条真实样品数据**：

| 数据源 | 样本数 | 可用参数 |
|--------|--------|----------|
| 柴达木盆地阿尔金山前带 | 16 | δ²H, δ¹³C, H₂%, CH₄% (最全) |
| 松辽盆地松科2井 | 30 | H₂%, CH₄%, C₂+, CO₂, N₂, He |
| 蛇绿岩杂岩渗漏 | 27 | H₂ Mol%, CH₄ Mol% |
| 中国天然氢显示统计 | 18 | δD 范围, H₂% |
| Kansas盆地Scott/Heins井 | 7 | H₂%, CH₄% |
| Amadeus盆地 | 4 | H₂%, CH₄%, R/Ra |

**参数缺失率**：R/Ra 100% 缺失（文献数据极少包含），δ¹³C-CH₄ 87% 缺失，δD 81% 缺失。缺失值以该成因类别的合成数据中位数填充。这些数据仅占合并数据集的 4.8%，不足以独立训练模型。

## 模型对比结果

`build_dataset.py` v2.0 在合成数据中引入了参数重叠和多构造背景混合，解决了 v1.0 "构造背景完美映射成因"的问题。

在 `hydrogen_realistic_dataset.csv` 上的 5 折交叉验证结果（9 模型）：

| 排名 | 模型 | 平均准确率 | 标准差 |
|------|------|-----------|--------|
| 1 | 随机森林 | 0.9965 | ±0.0025 |
| 2 | LightGBM | 0.9955 | ±0.0029 |
| 3 | XGBoost | 0.9945 | ±0.0029 |
| 4 | 梯度提升树 | 0.9940 | ±0.0041 |
| 5 | SVM (RBF) | 0.9935 | ±0.0041 |
| 6 | MLP | 0.9935 | ±0.0037 |
| 7 | 朴素贝叶斯 | 0.9935 | ±0.0041 |
| 8 | 逻辑回归 | 0.9925 | ±0.0016 |
| 9 | KNN (k=5) | 0.9925 | ±0.0042 |

**结论**：集成模型（随机森林、LightGBM、XGBoost）整体优于线性/距离模型。随机森林仍最优但优势不大。约 0.35%~0.75% 的误差来自参数重叠区的模糊样本，这正是模型对比的意义所在。

注意：合并了 102 条真实数据的 `hydrogen_combined_dataset.csv` 上模型仍然几乎满分——真实数据占比太小（4.8%），不足以产生挑战。

## Dependencies

`customtkinter`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`。全部列在 [requirements.txt](requirements.txt)。

## Architecture

### Data layer: [db_manager.py](db_manager.py)
- SQLite database (`hydrogen_system.db`) with two tables:
  - `training_data` — seeded from the CSV, used as an alternate data source for training.
  - `prediction_history` — every prediction made in the GUI is logged here for audit.
- WAL mode and foreign keys enabled on every connection.
- Each call to `_get_conn()` creates a new connection (not a singleton) for thread safety in the GUI context.

### Model: [train_model.py](train_model.py)
- Features: `delta_D`, `ln_CH4_H2`, `R_Ra`, `C1_C1_plus`, `delta_13C_CH4` (continuous) + `Tectonic_Setting` (one-hot encoded → 3 columns).
- Target: `Origin` with 4 classes — `壳源无机`, `幔源/蛇纹石化`, `有机质热解`, `微生物成因`.
- 80/20 stratified split, accuracy printed at end of training.

### GUI: [main_app.py](main_app.py)
- 3-tab `CTkTabview` with lazy-loading on tab switches (500ms polling monitor). Tabs:
  - **Tab 1 "AI 智能判识中心"** — 5 sliders + tectonic dropdown → predict button → displays predicted class + probability bars.
  - **Tab 2 "模型标准训练题库"** — `ttk.Treeview` with dark theme (`DarkTreeview` wrapper), loads CSV on first visit.
  - **Tab 3 "历史预测审计日志"** — reads from `prediction_history` table, double-click a row for detail popup.
- Model loaded at startup from `hydrogen_model.pkl`; if missing, the app runs but shows "未就绪" until a model is available.
- Prediction uses `pd.get_dummies` on the raw input dict, then `reindex` against the saved `feature_names` to handle missing columns.

### Scientific basis (from `C:\Users\。\Desktop\大创氢气\`)
The geochemical parameter ranges are derived from literature on natural hydrogen genesis:

| Parameter | Crustal Inorganic | Mantle/Serpentinization | Thermogenic | Microbial |
|-----------|-------------------|------------------------|-------------|-----------|
| δD (‰) | > -700 | < -700 (-850~-701) | -820~-640 | -830~-700 |
| ln(CH₄/H₂) | < -8 | > -4 | < -10 | ≥ 10 |
| R/Ra | < 0.1 | > 1 (up to 8) | < 0.1 | < 0.1 |
| C₁/C₁₊ | 0.95~0.99 | 0.99~1.0 | 0.80~0.95 | 0.99~1.0 |
| δ¹³C-CH₄ (‰) | -30~-10 | -25~0 | -50~-35 | -80~-55 |

The key discriminant is δD = -700‰ as the boundary between crustal (> -700) and mantle (< -700) sources. Microbial and serpentinization sources overlap significantly at δD < -700‰, requiring multi-parameter joint discrimination.

## Key technical details

- `hydrogen_model.pkl` 是一个字典 `{"model": <RandomForestClassifier>, "feature_names": [...]}`，不是裸模型对象。
- 独热编码顺序敏感：`pd.get_dummies` 按字母顺序生成列，应用预测时通过 `reindex` 对齐到保存的 `feature_names`。
- 数据库和 CSV 文件与脚本同目录，通过 `os.path.dirname(__file__)` 定位。
- GUI 通过 `.after(500)` 轮询标签页切换，以此实现惰性加载。
- `train_model.py` 和 `compare_models.py` 支持命令行参数指定数据集路径。
- `extract_real_data.py` 中的 `safe_float()` 能处理中文 Excel 中的范围格式（如 "−700~−830‰"）、缺失值占位符（"一"、"–"、"痕量"）等非标准格式。
- 尚无测试套件，未初始化版本控制。
- Windows 中文用户名路径会导致 `multiprocessing`（joblib）崩溃。所有脚本开头均已设置 `JOBLIB_TEMP_FOLDER` 到 `C:\Windows\Temp\sklearn` 以规避此问题。
