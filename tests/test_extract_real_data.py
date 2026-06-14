"""
测试 extract_real_data.py 中的核心函数：safe_float、assign_origin、assign_tectonic。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from extract_real_data import safe_float, assign_origin, assign_tectonic


class TestSafeFloat:
    """测试各种脏数据格式的解析能力。"""

    def test_int_and_float(self):
        assert safe_float(42) == 42.0
        assert safe_float(3.14) == 3.14

    def test_normal_string(self):
        assert safe_float("-722.5") == -722.5
        assert safe_float("0.437") == 0.437

    def test_chinese_missing_values(self):
        """中文缺失值占位符应返回 NaN。"""
        for val in ["一", "–", "—", "痕量", "-", ""]:
            assert np.isnan(safe_float(val))

    def test_english_missing_values(self):
        for val in ["n.d.", "<0.01"]:
            assert np.isnan(safe_float("<0.01"))  # 会被去前缀后尝试解析

    def test_range_format(self):
        """范围格式取中值。"""
        assert safe_float("-700~-830") == pytest.approx(-765.0)
        assert safe_float("−700~−830") == pytest.approx(-765.0)  # 中文减号
        assert safe_float("-687~-792") == pytest.approx(-739.5)

    def test_prefix_removal(self):
        """带前缀修饰符的值。"""
        assert safe_float(">99") == 99.0
        assert safe_float("<0.1") == 0.1
        assert safe_float("≥10") == 10.0
        assert safe_float("≈3.5") == 3.5

    def test_none_value(self):
        assert np.isnan(safe_float(None))


class TestAssignOrigin:
    """测试文献规则成因标注。"""

    def test_crustal_by_dD(self):
        """δD > -700 应判为壳源无机。"""
        row = {"delta_D": -650, "ln_CH4_H2": -9, "R_Ra": 0.05,
               "Tectonic_Setting": "活动断裂带"}
        assert assign_origin(row) == "壳源无机"

    def test_mantle_by_dD_and_tectonic(self):
        """δD < -700 + 超基性岩 → 幔源/蛇纹石化。"""
        row = {"delta_D": -780, "ln_CH4_H2": np.nan, "R_Ra": np.nan,
               "Tectonic_Setting": "超基性岩/洋壳"}
        assert assign_origin(row) == "幔源/蛇纹石化"

    def test_thermogenic_by_ln_ratio(self):
        """ln(CH₄/H₂) < -10 → 有机质热解。"""
        row = {"delta_D": np.nan, "ln_CH4_H2": -12.5, "R_Ra": np.nan,
               "Tectonic_Setting": "沉积盆地"}
        assert assign_origin(row) == "有机质热解"

    def test_microbial_by_ln_ratio(self):
        """ln(CH₄/H₂) ≥ 10 → 微生物成因。"""
        row = {"delta_D": np.nan, "ln_CH4_H2": 12.0, "R_Ra": np.nan,
               "Tectonic_Setting": "沉积盆地"}
        assert assign_origin(row) == "微生物成因"

    def test_R_Ra_dominant(self):
        """R/Ra ≥ 0.1 直接判幔源，覆盖其他参数。"""
        row = {"delta_D": -650, "ln_CH4_H2": -9, "R_Ra": 5.0,
               "Tectonic_Setting": "活动断裂带"}
        assert assign_origin(row) == "幔源/蛇纹石化"

    def test_mantle_by_tectonic_fallback(self):
        """无 δD 无 ln_ratio 时，蛇绿岩构造 → 幔源。"""
        row = {"delta_D": np.nan, "ln_CH4_H2": np.nan, "R_Ra": np.nan,
               "Tectonic_Setting": "超基性岩/洋壳"}
        assert assign_origin(row) == "幔源/蛇纹石化"

    def test_ln_ratio_mantle_range(self):
        """-4 < ln(CH₄/H₂) < 10 且无 δD → 幔源/蛇纹石化。"""
        row = {"delta_D": np.nan, "ln_CH4_H2": -3.0, "R_Ra": np.nan,
               "Tectonic_Setting": ""}
        assert assign_origin(row) == "幔源/蛇纹石化"


class TestAssignTectonic:
    """测试构造背景推断。"""

    def test_ophiolite_related(self):
        assert assign_tectonic("蛇绿岩") == "超基性岩/洋壳"
        assert assign_tectonic("橄榄岩") == "超基性岩/洋壳"
        assert assign_tectonic("超基性岩") == "超基性岩/洋壳"

    def test_basin_related(self):
        assert assign_tectonic("柴达木盆地") == "沉积盆地"
        assert assign_tectonic("松辽") == "沉积盆地"
        assert assign_tectonic("鄂尔多斯盆地") == "沉积盆地"

    def test_fault_volcanic(self):
        assert assign_tectonic("火山") == "活动断裂带"
        assert assign_tectonic("断裂") == "活动断裂带"
        assert assign_tectonic("基岩") == "活动断裂带"
