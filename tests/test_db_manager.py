"""
测试 db_manager.py 的数据库操作。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import shutil
from db_manager import _get_conn, init_db, log_prediction, get_prediction_history


@pytest.fixture(autouse=True)
def fresh_db():
    """每个测试前重置数据库。"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "hydrogen_system.db")
    # 备份原数据库
    backup = db_path + ".backup"
    if os.path.exists(db_path):
        shutil.copy2(db_path, backup)

    # 删除并重新初始化
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db()
    yield
    # 恢复原数据库
    if os.path.exists(backup):
        shutil.copy2(backup, db_path)
        os.remove(backup)


class TestDatabaseInit:
    """测试数据库初始化。"""

    def test_tables_exist(self):
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        conn.close()
        assert "training_data" in tables
        assert "prediction_history" in tables

    def test_wal_mode(self):
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"


class TestLogPrediction:
    """测试预测记录。"""

    def test_log_and_retrieve(self):
        input_data = {
            "delta_D": -750.0,
            "ln_CH4_H2": -3.5,
            "R_Ra": 5.2,
            "C1_C1_plus": 0.98,
            "delta_13C_CH4": -15.0,
            "Tectonic_Setting": "超基性岩/洋壳",
        }
        log_prediction(input_data, "幔源/蛇纹石化")

        df = get_prediction_history(limit=10)
        assert len(df) >= 1
        latest = df.iloc[0]
        assert latest["predicted_origin"] == "幔源/蛇纹石化"
        assert latest["delta_D"] == -750.0
        assert latest["R_Ra"] == 5.2
        assert latest["Tectonic_Setting"] == "超基性岩/洋壳"

    def test_multiple_predictions(self):
        for i in range(5):
            log_prediction({
                "delta_D": -700 + i,
                "ln_CH4_H2": -5.0,
                "R_Ra": 0.05,
                "C1_C1_plus": 0.95,
                "delta_13C_CH4": -20.0,
                "Tectonic_Setting": "活动断裂带",
            }, f"成因{i}")
        df = get_prediction_history(limit=100)
        assert len(df) == 5

    def test_limit(self):
        for i in range(20):
            log_prediction({
                "delta_D": -700.0,
                "ln_CH4_H2": 0.0,
                "R_Ra": 0.1,
                "C1_C1_plus": 1.0,
                "delta_13C_CH4": 0.0,
                "Tectonic_Setting": "沉积盆地",
            }, "test")
        df = get_prediction_history(limit=10)
        assert len(df) == 10  # 只取最近 10 条
