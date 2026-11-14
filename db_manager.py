"""
数据库管理模块 — SQLite 数据访问层
提供训练数据管理、预测历史记录两大核心功能。
"""
import os
import sqlite3
import sys

import pandas as pd


def _data_dir() -> str:
    """返回可持久化的数据目录。

    源码运行时使用脚本所在目录；PyInstaller 打包后 __file__ 指向临时解压
    目录（%TEMP%\_MEIxxxx），退出即被删除，因此改用 LOCALAPPDATA。
    """
    if getattr(sys, "frozen", False):
        base = os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "HydrogenSystem",
        )
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base, exist_ok=True)
    return base


DB_PATH = os.path.join(_data_dir(), "hydrogen_system.db")


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接（每次调用创建新连接，线程安全）。"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")       # 并发读写性能
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
    except sqlite3.Error as e:
        raise RuntimeError(f"数据库连接失败: {e}")


# ═══════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════

def init_db():
    """创建 training_data 和 prediction_history 两张表（如不存在）。"""
    conn = _get_conn()
    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                delta_D         REAL    NOT NULL,
                ln_CH4_H2       REAL    NOT NULL,
                R_Ra            REAL    NOT NULL,
                C1_C1_plus      REAL    NOT NULL,
                delta_13C_CH4   REAL    NOT NULL,
                Tectonic_Setting TEXT   NOT NULL,
                Origin          TEXT    NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                delta_D         REAL    NOT NULL,
                ln_CH4_H2       REAL    NOT NULL,
                R_Ra            REAL    NOT NULL,
                C1_C1_plus      REAL    NOT NULL,
                delta_13C_CH4   REAL    NOT NULL,
                Tectonic_Setting TEXT   NOT NULL,
                predicted_origin TEXT   NOT NULL
            )
        """)

        conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"数据库初始化失败: {e}")
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════
# 训练数据 (表 A)
# ═══════════════════════════════════════════════════════════════

def seed_training_data():
    """若 training_data 表为空，则从 CSV 数据集文件读取数据并填充。"""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM training_data")
        count = cur.fetchone()[0]

        if count > 0:
            return  # 已有数据，跳过

        csv_path = os.path.join(os.path.dirname(__file__), "hydrogen_realistic_dataset.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"数据集文件未找到: {csv_path}，请先运行 build_dataset.py 生成数据集。"
            )

        df = pd.read_csv(csv_path)

        rows = []
        for _, row in df.iterrows():
            rows.append((
                row["delta_D"], row["ln_CH4_H2"], row["R_Ra"],
                row["C1_C1_plus"], row["delta_13C_CH4"],
                row["Tectonic_Setting"], row["Origin"],
            ))

        cur.executemany(
            """INSERT INTO training_data
               (delta_D, ln_CH4_H2, R_Ra, C1_C1_plus, delta_13C_CH4, Tectonic_Setting, Origin)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"训练数据写入失败: {e}")
    finally:
        conn.close()


def get_training_data() -> pd.DataFrame:
    """从 training_data 表读取全部数据，返回 DataFrame（供模型训练使用）。"""
    conn = _get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT delta_D, ln_CH4_H2, R_Ra, C1_C1_plus, "
            "       delta_13C_CH4, Tectonic_Setting, Origin "
            "FROM training_data",
            conn,
        )
        return df
    except sqlite3.Error as e:
        raise RuntimeError(f"训练数据读取失败: {e}")
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════
# 预测历史 (表 B)
# ═══════════════════════════════════════════════════════════════

def log_prediction(input_dict: dict, result: str):
    """将一次用户预测记录写入 prediction_history 表。

    Args:
        input_dict: 包含 delta_D, ln_CH4_H2, R_Ra, C1_C1_plus,
                    delta_13C_CH4, Tectonic_Setting 的字典。
        result:     模型预测的成因名称（如 "壳源无机"）。
    """
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO prediction_history
               (delta_D, ln_CH4_H2, R_Ra, C1_C1_plus, delta_13C_CH4, Tectonic_Setting, predicted_origin)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                input_dict["delta_D"],
                input_dict["ln_CH4_H2"],
                input_dict["R_Ra"],
                input_dict["C1_C1_plus"],
                input_dict["delta_13C_CH4"],
                input_dict["Tectonic_Setting"],
                result,
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"预测记录写入失败: {e}")
    finally:
        conn.close()


def get_prediction_history(limit: int = 100) -> pd.DataFrame:
    """查询 prediction_history 表，按时间倒序返回最近 N 条记录。"""
    conn = _get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT id, timestamp, delta_D, ln_CH4_H2, R_Ra, C1_C1_plus, "
            "       delta_13C_CH4, Tectonic_Setting, predicted_origin "
            "FROM prediction_history "
            "ORDER BY id DESC "
            "LIMIT ?",
            conn,
            params=(limit,),
        )
        return df
    except sqlite3.Error as e:
        raise RuntimeError(f"历史记录查询失败: {e}")
    finally:
        conn.close()
