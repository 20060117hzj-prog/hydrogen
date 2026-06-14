"""
测试 build_dataset.py 的数据生成逻辑。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from build_dataset import build_dataset, CLASS_DEFS, N_PER_CLASS


class TestBuildDataset:
    """测试合成数据集的结构和内容。"""

    @pytest.fixture
    def df(self):
        return build_dataset()

    def test_total_rows(self, df):
        """应生成 4×500=2000 条数据。"""
        assert len(df) == 2000

    def test_columns(self, df):
        expected = ["delta_D", "ln_CH4_H2", "R_Ra", "C1_C1_plus",
                    "delta_13C_CH4", "Tectonic_Setting", "Origin"]
        assert list(df.columns) == expected

    def test_class_balance(self, df):
        """四类各 500 条。"""
        counts = df["Origin"].value_counts()
        expected = ["壳源无机", "幔源/蛇纹石化", "有机质热解", "微生物成因"]
        for cls in expected:
            assert counts[cls] == 500

    def test_tectonic_not_single_per_class(self, df):
        """v2.0: 每类成因应有多种构造背景。"""
        for origin in df["Origin"].unique():
            tectonics = df[df["Origin"] == origin]["Tectonic_Setting"].unique()
            assert len(tectonics) >= 2, f"{origin} 的构造背景种类 < 2"

    def test_no_missing_values(self, df):
        """不应有任何缺失值。"""
        assert df.isna().sum().sum() == 0

    def test_delta_D_overlap_exists(self, df):
        """δD 在幔源和微生物成因之间应有重叠。"""
        mantle = df[df["Origin"] == "幔源/蛇纹石化"]["delta_D"]
        microbial = df[df["Origin"] == "微生物成因"]["delta_D"]
        lo = max(mantle.min(), microbial.min())
        hi = min(mantle.max(), microbial.max())
        assert lo < hi, f"δD 无重叠区: mantle=[{mantle.min():.0f}, {mantle.max():.0f}], microbial=[{microbial.min():.0f}, {microbial.max():.0f}]"
        # 重叠区内应有相当数量的样本
        n_overlap = len(df[df["Origin"].isin(["幔源/蛇纹石化", "微生物成因"])
                         & df["delta_D"].between(lo, hi)])
        assert n_overlap > 500, f"δD 重叠区仅 {n_overlap} 条"

    def test_parameter_ranges_within_class_def(self, df):
        """所有参数值应在其类定义的范围内。"""
        num_params = ["delta_D", "ln_CH4_H2", "R_Ra", "C1_C1_plus", "delta_13C_CH4"]
        for origin, _, ranges in CLASS_DEFS:
            subset = df[df["Origin"] == origin]
            for param in num_params:
                lo, hi = ranges[param]
                assert subset[param].min() >= lo, \
                    f"{origin}.{param} min({subset[param].min():.2f}) < {lo}"
                assert subset[param].max() <= hi, \
                    f"{origin}.{param} max({subset[param].max():.2f}) > {hi}"

    def test_reproducibility(self):
        """两次生成应完全相同（固定种子）。"""
        df1 = build_dataset()
        df2 = build_dataset()
        assert df1.equals(df2)
