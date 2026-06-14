"""
模型对比脚本 — 对9种分类器进行5折交叉验证，横向评估在氢气成因分类任务上的表现。

输出：
  - 终端：准确率排名表（均值 ± 标准差）、训练时间对比、最优模型推荐
  - 图片：model_comparison_confusion.png（9个模型的混淆矩阵）
"""
import os
import sys
import time
import warnings

# 强制 UTF-8 输出（Windows GBK 兼容）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # 中文用户名路径会导致 multiprocessing 崩溃，指定纯 ASCII 临时目录
    _temp_dir = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Temp", "sklearn")
    os.makedirs(_temp_dir, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = _temp_dir
    os.environ["TMPDIR"] = _temp_dir
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
# 支持命令行指定数据集路径
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
else:
    DATASET_PATH = os.path.join(BASE_DIR, "hydrogen_realistic_dataset.csv")
# 输出文件名跟随输入文件名
dataset_tag = os.path.splitext(os.path.basename(DATASET_PATH))[0]
OUTPUT_FIG = os.path.join(BASE_DIR, f"model_comparison_{dataset_tag}.png")

# ── 模型定义 ───────────────────────────────────────────
# 需要缩放的模型用 Pipeline 包裹
def build_models():
    return {
        "随机森林": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "逻辑回归": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42, probability=True)),
        ]),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "梯度提升树": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=150, random_state=42, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=150, random_state=42, verbose=-1),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,
                                  random_state=42)),
        ]),
        "朴素贝叶斯": GaussianNB(),
    }


# ── 主流程 ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  氢气成因分类 — 9 模型对比评估（5 折交叉验证）")
    print(f"  数据集: {os.path.basename(DATASET_PATH)}")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=["Origin"])
    y_raw = df["Origin"]
    X_encoded = pd.get_dummies(X, columns=["Tectonic_Setting"])
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    print(f"  样本数: {len(df)}, 特征数: {X_encoded.shape[1]}")
    print(f"  类别: {list(class_names)}")

    # 2. 交叉验证
    print("\n[2/3] 5 折交叉验证中...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()

    results = []
    for name, model in models.items():
        print(f"  [{name}] ...", end=" ", flush=True)

        t0 = time.perf_counter()
        scores = cross_validate(
            model, X_encoded, y,
            cv=cv, scoring="accuracy",
            return_train_score=False, n_jobs=-1,
        )
        elapsed = time.perf_counter() - t0

        test_scores = scores["test_score"]
        results.append({
            "模型": name,
            "平均准确率": np.mean(test_scores),
            "标准差": np.std(test_scores),
            "最低": np.min(test_scores),
            "最高": np.max(test_scores),
            "总训练时间(s)": elapsed,
        })
        print(f"  {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

    # 3. 输出排名表
    results.sort(key=lambda r: r["平均准确率"], reverse=True)

    print("\n" + "=" * 70)
    print("  模型排名（按 5 折平均准确率降序）")
    print("=" * 70)
    header = f"{'排名':<4} {'模型':<14} {'平均准确率':<10} {'标准差':<8} {'最低':<8} {'最高':<8} {'训练时间':<10}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        print(f"{i:<4} {r['模型']:<14} {r['平均准确率']:<10.4f} "
              f"{r['标准差']:<8.4f} {r['最低']:<8.4f} {r['最高']:<8.4f} "
              f"{r['总训练时间(s)']:<10.2f}")

    # 4. 每类指标 + 混淆矩阵 (全量训练)
    print("\n[3/3] 生成混淆矩阵 & 分类报告...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        model_clone = build_models()[name]
        model_clone.fit(X_encoded, y)
        y_pred = model_clone.predict(X_encoded)

        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx], cbar=False, square=True)
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("预测")
        axes[idx].set_ylabel("真实")

    fig.suptitle("氢气成因分类 — 9 模型混淆矩阵对比", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩阵已保存至: {OUTPUT_FIG}")

    # 5. 最优模型详细报告
    best_name = results[0]["模型"]
    best_model = build_models()[best_name]
    best_model.fit(X_encoded, y)
    y_pred_best = best_model.predict(X_encoded)

    print(f"\n{'=' * 70}")
    print(f"  最优模型: {best_name}  详细分类报告")
    print(f"{'=' * 70}")
    print(classification_report(y, y_pred_best, target_names=class_names, digits=4))

    # 检查是否存在多种模型并列满分
    perfect_models = [r["模型"] for r in results if r["平均准确率"] >= 0.9995]
    if len(perfect_models) >= 2:
        print(f"  注意: {len(perfect_models)} 个模型均达到 ~100% 准确率 ({', '.join(perfect_models)})。")
        print(f"  这说明合成数据类别间完全可分，建议引入真实数据或参数重叠后再做对比。")
        print(f"  当前随机森林已足够胜任此任务。")
    else:
        rf_score = [r for r in results if r["模型"] == "随机森林"][0]["平均准确率"]
        print(f"  推荐: 使用【{best_name}】替代当前随机森林，"
              f"准确率从 {rf_score:.4f} → {results[0]['平均准确率']:.4f}")

    print("\n" + "=" * 70)
    print("  对比完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
