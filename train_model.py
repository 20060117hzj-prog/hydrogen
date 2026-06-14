"""
模型训练脚本 — 从 CSV 数据集读取数据，训练 RandomForest 并持久化模型。

用法:
    python train_model.py                           # 用默认合成数据集
    python train_model.py hydrogen_combined_dataset.csv  # 用合并数据集
"""
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
    if not os.path.isabs(DATASET_PATH):
        DATASET_PATH = os.path.join(BASE_DIR, DATASET_PATH)
else:
    DATASET_PATH = os.path.join(BASE_DIR, "hydrogen_realistic_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "hydrogen_model.pkl")


def main():
    print("=" * 60)
    print("  氢气成因多维智能判识 — 模型训练")
    print("=" * 60)

    # 1. 检查数据集是否存在
    print("\n[1/4] 检查标准数据集...")
    if not os.path.exists(DATASET_PATH):
        print(f"  错误: 未找到 {DATASET_PATH}")
        print("  请先运行 build_dataset.py 生成数据集。")
        return
    print(f"  数据集: {os.path.basename(DATASET_PATH)} ({DATASET_PATH})")

    # 2. 加载数据
    print("\n[2/4] 从 CSV 加载训练数据...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  数据量: {len(df)} 条, 类别分布:")
    for label, count in df["Origin"].value_counts().items():
        print(f"    {label}: {count}")

    # 3. 预处理 + 训练
    print("\n[3/4] 数据预处理 & 训练 RandomForest...")
    X = df.drop(columns=["Origin"])
    y = df["Origin"]

    X_encoded = pd.get_dummies(X, columns=["Tectonic_Setting"])
    feature_names = X_encoded.columns.tolist()

    print(f"  独热编码后特征维度: {len(feature_names)}")
    for name in feature_names:
        print(f"    - {name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  测试集准确率: {acc:.4f} ({acc * 100:.2f}%)")

    # 4. 保存模型
    print("\n[4/4] 保存模型...")
    joblib.dump({"model": model, "feature_names": feature_names}, MODEL_PATH)
    print(f"  模型已保存至: {MODEL_PATH}")
    print("\n" + "=" * 60)
    print("  训练完成！可运行 main_app.py 启动桌面应用。")
    print("=" * 60)


if __name__ == "__main__":
    main()
