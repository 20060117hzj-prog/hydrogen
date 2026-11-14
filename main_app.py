# -*- coding: utf-8 -*-
"""氢气成因多维智能判识系统 V2.0

桌面客户端：三页式界面（智能判识 / 训练题库 / 审计日志）。
界面层与模型层、数据层解耦，模型文件与数据集与程序同目录。
"""
import os
from tkinter import ttk, messagebox

import customtkinter as ctk
import joblib
import pandas as pd

from db_manager import init_db, log_prediction, get_prediction_history

# ── 主题 ───────────────────────────────────────────────────
BG = "#0E1317"
HEADER_BG = "#121A20"
CARD = "#161E25"
CARD_2 = "#1B242C"
BORDER = "#24313A"
TEXT = "#E8EFF5"
MUTED = "#8C9BA8"
ACCENT = "#2FD3C0"
DANGER = "#E8836B"
OK = "#57C98B"

CLASS_COLORS = {
    "壳源无机": "#5AA9FF",
    "幔源/蛇纹石化": "#2FD3C0",
    "有机质热解": "#F5A45B",
    "微生物成因": "#B489FF",
}

FONT = "Microsoft YaHei UI"


def f(size, bold=False):
    return ctk.CTkFont(family=FONT, size=size, weight="bold" if bold else "normal")


# ── 路径 ───────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hydrogen_model.pkl")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "hydrogen_realistic_dataset.csv")

# ── 参数定义 ───────────────────────────────────────────────
SLIDER_DEFS = [
    ("δD (‰)", -850.0, -100.0, -500.0, ".0f", "氢同位素组成"),
    ("ln(CH₄/H₂)", -15.0, 15.0, -5.0, ".1f", "甲烷与氢气比的对数"),
    ("R/Ra", 0.01, 8.0, 1.0, ".2f", "氦同位素比值"),
    ("C₁/C₁₊", 0.80, 1.0, 0.95, ".3f", "甲烷占烷烃比例"),
    ("δ¹³C-CH₄ (‰)", -80.0, 0.0, -40.0, ".1f", "甲烷碳同位素"),
]

TECTONIC_OPTIONS = ["活动断裂带", "超基性岩/洋壳", "沉积盆地"]

TABS = ["AI 智能判识中心", "模型标准训练题库", "历史预测审计日志"]


# ═══════════════════════════════════════════════════════════
# 通用组件
# ═══════════════════════════════════════════════════════════
class Card(ctk.CTkFrame):
    """带描边的卡片容器。"""

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("fg_color", CARD)
        kwargs.setdefault("corner_radius", 14)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", BORDER)
        super().__init__(parent, **kwargs)


class SectionTitle(ctk.CTkFrame):
    """左侧带竖条的区块标题。"""

    def __init__(self, parent, title, subtitle=""):
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)

        bar = ctk.CTkFrame(self, width=3, height=18, corner_radius=2, fg_color=ACCENT)
        bar.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 10))
        bar.grid_propagate(False)

        ctk.CTkLabel(self, text=title, font=f(14, True), text_color=TEXT,
                     anchor="w").grid(row=0, column=1, sticky="w")
        if subtitle:
            ctk.CTkLabel(self, text=subtitle, font=f(10), text_color=MUTED,
                         anchor="w").grid(row=1, column=1, sticky="w", pady=(1, 0))


class SliderRow(ctk.CTkFrame):
    """参数名 + 说明 + 数值胶囊 + 滑块。"""

    def __init__(self, parent, label, frm, to, default, fmt, hint):
        super().__init__(parent, fg_color="transparent")
        self.fmt = fmt
        self.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        head.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(head, text=label, font=f(12, True), text_color=TEXT,
                     anchor="w").grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(head, text=hint, font=f(8), text_color="#5A6772",
                     anchor="w").grid(row=0, column=2, sticky="w", padx=(10, 8))

        self.chip = ctk.CTkLabel(head, text=f"{default:{fmt}}", width=72, height=24,
                                 corner_radius=6, fg_color=CARD_2, text_color=ACCENT,
                                 font=("Consolas", 12))
        self.chip.grid(row=0, column=3, sticky="e")

        self.slider = ctk.CTkSlider(
            self, from_=frm, to=to, command=self._on_change,
            height=16, corner_radius=8,
            fg_color=CARD_2, progress_color=ACCENT,
            button_color=TEXT, button_hover_color="#FFFFFF",
            button_length=16,
        )
        self.slider.set(default)
        self.slider.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    def _on_change(self, val):
        self.chip.configure(text=f"{val:{self.fmt}}")

    def get(self):
        return self.slider.get()

    def reset(self, value):
        self.slider.set(value)
        self._on_change(value)


class ProbRow(ctk.CTkFrame):
    """一行概率：类别名 + 进度条 + 百分比。"""

    def __init__(self, parent, name, color):
        super().__init__(parent, fg_color="transparent")
        self.color = color
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text=name, width=108, anchor="w",
                     font=f(11), text_color=MUTED).grid(row=0, column=0, sticky="w")

        self.bar = ctk.CTkProgressBar(self, height=10, corner_radius=5,
                                      fg_color="#26333D", progress_color=color)
        self.bar.set(0)
        self.bar.grid(row=0, column=1, sticky="ew", padx=(6, 12))

        self.pct = ctk.CTkLabel(self, text="—", width=52, anchor="e",
                                font=("Consolas", 11), text_color=TEXT)
        self.pct.grid(row=0, column=2, sticky="e")

    def set(self, value):
        # 概率为 0 时把进度条染成底色，避免在左端留下一个彩色圆点
        self.bar.configure(progress_color=self.color if value >= 0.005 else "#26333D")
        self.bar.set(value)
        self.pct.configure(text=f"{value * 100:5.1f}%")


class DataTable(ctk.CTkFrame):
    """暗色 Treeview + 滚动条。"""

    _style_ready = False

    def __init__(self, parent, columns, headings, widths, anchor="center"):
        super().__init__(parent, fg_color="transparent")
        self._init_style()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.vbar = ctk.CTkScrollbar(self, orientation="vertical", width=12,
                                     fg_color="transparent", button_color="#2C3A44",
                                     button_hover_color="#3A4C58")

        self.tree = ttk.Treeview(self, columns=columns, show="headings",
                                 style="App.Treeview",
                                 yscrollcommand=self.vbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.vbar.configure(command=self.tree.yview)

        for col, head, width in zip(columns, headings, widths):
            self.tree.heading(col, text=head)
            self.tree.column(col, width=width, minwidth=48, anchor=anchor, stretch=True)

        self.tree.tag_configure("odd", background=CARD)
        self.tree.tag_configure("even", background="#141B22")

    @classmethod
    def _init_style(cls):
        if cls._style_ready:
            return
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "App.Treeview",
            background=CARD, fieldbackground=CARD, foreground=TEXT,
            borderwidth=0, rowheight=30, font=(FONT, 9),
        )
        style.configure(
            "App.Treeview.Heading",
            background=CARD_2, foreground=MUTED, borderwidth=0,
            relief="flat", font=(FONT, 9, "bold"), padding=(6, 8),
        )
        style.layout("App.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.map("App.Treeview.Heading", background=[("active", "#222D36")])
        style.map("App.Treeview",
                  background=[("selected", "#24404A")],
                  foreground=[("selected", "#FFFFFF")])
        cls._style_ready = True

    def clear(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def insert_row(self, values, index=0):
        tag = "even" if index % 2 else "odd"
        return self.tree.insert("", "end", values=values, tags=(tag,))


# ═══════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        self.title("氢气成因多维智能判识系统")
        self.geometry("1160x780")
        self.minsize(1000, 700)
        self.configure(fg_color=BG)

        self.model = None
        self.feature_names = None
        self.model_loaded = False
        self._training_loaded = False
        self._current = TABS[0]

        try:
            init_db()
        except Exception as e:
            messagebox.showwarning("数据库错误", f"无法初始化数据库:\n{e}")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_tabs()
        self._build_content()
        self._build_status()

        self._try_load_model()
        self._show_tab(TABS[0])

    # ── 顶部 ─────────────────────────────────────────────
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=HEADER_BG, corner_radius=0, height=76)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        mark = ctk.CTkFrame(header, width=4, height=28, corner_radius=2, fg_color=ACCENT)
        mark.grid(row=0, column=0, rowspan=2, padx=(24, 14), pady=(18, 0), sticky="nw")
        mark.grid_propagate(False)

        ctk.CTkLabel(header, text="氢气成因多维智能判识系统", font=f(19, True),
                     text_color=TEXT, anchor="w").grid(row=0, column=1, sticky="sw", pady=(18, 0))

        sub = ctk.CTkFrame(header, fg_color="transparent")
        sub.grid(row=1, column=1, sticky="nw")
        ctk.CTkLabel(sub, text="Hydrogen Genesis Intelligent Identifier", font=f(9),
                     text_color="#5E6C77").pack(side="left")
        ctk.CTkLabel(sub, text="  ·  ", font=f(9), text_color="#3A4750").pack(side="left")
        ctk.CTkLabel(sub, text="随机森林四分类", font=f(9),
                     text_color="#5E6C77").pack(side="left")

        right = ctk.CTkFrame(header, fg_color="transparent")
        right.grid(row=0, column=2, rowspan=2, padx=(0, 24), pady=(18, 0), sticky="ne")

        self.status_pill = ctk.CTkFrame(right, fg_color=CARD, corner_radius=14,
                                        border_width=1, border_color=BORDER, height=28)
        self.status_pill.pack(side="left", padx=(0, 10))
        self.pill_dot = ctk.CTkLabel(self.status_pill, text="●", font=f(9), text_color=MUTED,
                                     width=10)
        self.pill_dot.pack(side="left", padx=(10, 4), pady=3)
        self.pill_text = ctk.CTkLabel(self.status_pill, text="初始化", font=f(10),
                                      text_color=MUTED)
        self.pill_text.pack(side="left", padx=(0, 10), pady=3)

        ctk.CTkLabel(right, text="V2.0", font=("Consolas", 11, "bold"), text_color=ACCENT,
                     width=48, height=26, corner_radius=6,
                     fg_color="#12302D").pack(side="left")

        ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0).grid(
            row=0, column=0, sticky="sew")

    def _build_tabs(self):
        bar = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        bar.grid(row=1, column=0, sticky="ew", padx=24, pady=(14, 4))
        bar.grid_columnconfigure(1, weight=1)

        self.segment = ctk.CTkSegmentedButton(
            bar, values=TABS, command=self._show_tab,
            font=f(12), height=34, corner_radius=17,
            fg_color=CARD, selected_color=ACCENT, selected_hover_color="#3BE0CE",
            unselected_color=CARD, unselected_hover_color=CARD_2,
            text_color=TEXT,
        )
        self.segment.set(TABS[0])
        self.segment.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(bar, text="本地运行 · 数据不出本机", font=f(9),
                     text_color="#4E5B66").grid(row=0, column=2, sticky="e")

    # ── 内容区 ───────────────────────────────────────────
    def _build_content(self):
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=2, column=0, sticky="nsew", padx=24, pady=(6, 10))
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.views = {
            TABS[0]: self._build_predict_view(),
            TABS[1]: self._build_bank_view(),
            TABS[2]: self._build_log_view(),
        }
        for view in self.views.values():
            view.grid(row=0, column=0, sticky="nsew")

    # 视图一：判识
    def _build_predict_view(self):
        view = ctk.CTkFrame(self.content, fg_color="transparent")
        view.grid_columnconfigure(0, weight=0, minsize=392)
        view.grid_columnconfigure(1, weight=1)
        view.grid_rowconfigure(0, weight=1)

        left = Card(view)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.grid_columnconfigure(0, weight=1)

        SectionTitle(left, "地质参数输入", "五项地球化学指标与构造背景").grid(
            row=0, column=0, sticky="ew", padx=18, pady=(16, 10))

        ctk.CTkLabel(left, text="构造背景", font=f(11, True), text_color="#B9C6D0",
                     anchor="w").grid(row=1, column=0, sticky="w", padx=18)

        self.tectonic_var = ctk.StringVar(value=TECTONIC_OPTIONS[0])
        self.tectonic = ctk.CTkOptionMenu(
            left, values=TECTONIC_OPTIONS, variable=self.tectonic_var,
            font=f(12), height=36, corner_radius=8,
            fg_color=CARD_2, button_color="#2A3A44", button_hover_color="#354854",
            dropdown_fg_color=CARD_2, dropdown_hover_color="#2A3A44",
            dropdown_text_color=TEXT, text_color=TEXT,
        )
        self.tectonic.grid(row=2, column=0, sticky="ew", padx=18, pady=(6, 12))

        self.sliders = []
        for i, (label, frm, to, dft, fmt, hint) in enumerate(SLIDER_DEFS):
            row = SliderRow(left, label, frm, to, dft, fmt, hint)
            row.grid(row=3 + i, column=0, sticky="ew", padx=18, pady=(0, 6))
            self.sliders.append(row)

        ctk.CTkButton(left, text="恢复默认值", command=self._reset_inputs,
                      font=f(10), height=30, corner_radius=8,
                      fg_color="transparent", hover_color=CARD_2,
                      border_width=1, border_color=BORDER, text_color=MUTED).grid(
            row=9, column=0, sticky="ew", padx=18, pady=(4, 12))

        ctk.CTkLabel(left, text="提示：R/Ra 缺失会明显降低判识可靠性；单位与基准请与测试报告保持一致。",
                     font=f(9), text_color="#55626D", anchor="w", justify="left",
                     wraplength=330).grid(row=10, column=0, sticky="w", padx=18,
                                          pady=(0, 14))

        right = Card(view)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)
        right.grid_rowconfigure(5, weight=0)

        SectionTitle(right, "智能判识结果", "模型输出成因类别与概率分布").grid(
            row=0, column=0, sticky="ew", padx=18, pady=(16, 10))

        self.btn_predict = ctk.CTkButton(
            right, text="启动 AI 判识", command=self._on_predict,
            font=f(15, True), height=46, corner_radius=10,
            fg_color=ACCENT, hover_color="#3BE0CE", text_color="#06201D",
        )
        self.btn_predict.grid(row=1, column=0, sticky="ew", padx=18)

        result_box = ctk.CTkFrame(right, fg_color=CARD_2, corner_radius=12,
                                  border_width=1, border_color=BORDER)
        result_box.grid(row=2, column=0, sticky="ew", padx=18, pady=10)
        result_box.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(result_box, text="最可能成因", font=f(10), text_color=MUTED,
                     anchor="w").grid(row=0, column=0, sticky="w", padx=16, pady=(10, 0))

        self.result_label = ctk.CTkLabel(result_box, text="等待判识", font=f(28, True),
                                         text_color="#4E5B66", anchor="w")
        self.result_label.grid(row=1, column=0, sticky="w", padx=16)

        self.conf_label = ctk.CTkLabel(result_box, text="输入参数后点击上方按钮", font=f(10),
                                       text_color=MUTED, anchor="w")
        self.conf_label.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 10))

        prob_card = ctk.CTkFrame(right, fg_color="transparent")
        prob_card.grid(row=3, column=0, sticky="nsew", padx=18, pady=(0, 8))
        prob_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(prob_card, text="各类成因概率", font=f(11, True),
                     text_color="#B9C6D0", anchor="w").grid(row=0, column=0, sticky="w",
                                                            pady=(0, 6))

        self.prob_rows = {}
        for i, name in enumerate(["壳源无机", "幔源/蛇纹石化", "有机质热解", "微生物成因"]):
            row = ProbRow(prob_card, name, CLASS_COLORS[name])
            row.grid(row=1 + i, column=0, sticky="ew", pady=2)
            self.prob_rows[name] = row

        ctk.CTkFrame(right, fg_color=BORDER, height=1, corner_radius=0).grid(
            row=4, column=0, sticky="ew", padx=18, pady=(2, 10))

        ctk.CTkLabel(
            right,
            text="提示：最高概率低于 60% 时，样品多处于成因过渡区，建议补充 R/Ra 与 δ¹³C-CH₄ 数据后复核。",
            font=f(9), text_color="#55626D", anchor="w", justify="left",
            wraplength=640).grid(row=5, column=0, sticky="w", padx=18)

        ctk.CTkLabel(right, text="随机森林 · 150 棵树　|　判识结果自动写入本地审计日志", font=f(9),
                     text_color="#4E5B66").grid(row=6, column=0, sticky="w", padx=18,
                                                pady=(8, 14))
        return view

    # 视图二：题库
    def _build_bank_view(self):
        view = ctk.CTkFrame(self.content, fg_color="transparent")
        view.grid_rowconfigure(0, weight=1)
        view.grid_columnconfigure(0, weight=1)

        card = Card(view)
        card.grid(row=0, column=0, sticky="nsew")
        card.grid_rowconfigure(2, weight=1)
        card.grid_columnconfigure(0, weight=1)

        SectionTitle(card, "模型标准训练题库", "2000 条标准样本 · 四类成因各 500 条").grid(
            row=0, column=0, sticky="ew", padx=18, pady=(18, 8))

        self.bank_desc = ctk.CTkLabel(
            card, text="首次进入该页时自动加载数据集，请稍候…", font=f(10),
            text_color=MUTED, anchor="w", justify="left")
        self.bank_desc.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))

        cols = ["delta_D", "ln_CH4_H2", "R_Ra", "C1_C1_plus", "delta_13C_CH4",
                "Tectonic_Setting", "Origin"]
        heads = ["δD (‰)", "ln(CH₄/H₂)", "R/Ra", "C₁/C₁₊", "δ¹³C-CH₄ (‰)", "构造背景", "成因类型"]
        widths = [90, 102, 80, 86, 110, 134, 128]
        self.bank_table = DataTable(card, cols, heads, widths)
        self.bank_table.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        return view

    # 视图三：日志
    def _build_log_view(self):
        view = ctk.CTkFrame(self.content, fg_color="transparent")
        view.grid_rowconfigure(0, weight=1)
        view.grid_columnconfigure(0, weight=1)

        card = Card(view)
        card.grid(row=0, column=0, sticky="nsew")
        card.grid_rowconfigure(2, weight=1)
        card.grid_columnconfigure(0, weight=1)

        SectionTitle(card, "历史预测审计日志", "按时间倒序 · 双击查看明细").grid(
            row=0, column=0, sticky="ew", padx=18, pady=(18, 8))

        toolbar = ctk.CTkFrame(card, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))
        toolbar.grid_columnconfigure(0, weight=1)

        self.log_desc = ctk.CTkLabel(toolbar, text="暂无记录", font=f(10),
                                     text_color=MUTED, anchor="w")
        self.log_desc.grid(row=0, column=0, sticky="w")

        ctk.CTkButton(toolbar, text="刷新", command=self._refresh_history,
                      font=f(10), width=64, height=28, corner_radius=8,
                      fg_color=CARD_2, hover_color="#2A3A44", text_color=TEXT).grid(
            row=0, column=1, sticky="e")

        cols = ["id", "timestamp", "delta_D", "ln_CH4_H2", "R_Ra", "C1_C1_plus",
                "delta_13C_CH4", "Tectonic_Setting", "predicted_origin"]
        heads = ["#", "判识时间", "δD (‰)", "ln(CH₄/H₂)", "R/Ra", "C₁/C₁₊",
                 "δ¹³C-CH₄ (‰)", "构造背景", "AI 结论"]
        widths = [50, 154, 86, 98, 74, 82, 106, 126, 118]
        self.log_table = DataTable(card, cols, heads, widths)
        self.log_table.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.log_table.tree.bind("<Double-1>", self._on_row_detail)
        return view

    def _build_status(self):
        bar = ctk.CTkFrame(self, fg_color=HEADER_BG, corner_radius=0, height=34)
        bar.grid(row=3, column=0, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        self.status_dot = ctk.CTkLabel(bar, text="●", font=f(9), text_color=MUTED, width=10)
        self.status_dot.grid(row=0, column=0, padx=(24, 6), pady=6)

        self.status_label = ctk.CTkLabel(bar, text="准备就绪", font=f(10),
                                         text_color=MUTED, anchor="w")
        self.status_label.grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(bar, text="SQLite · 本地存储", font=f(9), text_color="#4E5B66").grid(
            row=0, column=2, sticky="e", padx=(0, 24))

    # ── 状态与提示 ───────────────────────────────────────
    def _set_status(self, text, color=MUTED):
        self.status_label.configure(text=text, text_color=color)
        self.status_dot.configure(text_color=color)

    def _set_pill(self, text, color=MUTED):
        self.pill_text.configure(text=text, text_color=color)
        self.pill_dot.configure(text_color=color)

    # ── 标签切换 ─────────────────────────────────────────
    def _show_tab(self, name):
        self._current = name
        if self.segment.get() != name:
            self.segment.set(name)
        self.views[name].tkraise()
        if name == TABS[1]:
            self._load_training_data()
        elif name == TABS[2]:
            self._refresh_history()

    # ── 模型 ─────────────────────────────────────────────
    def _try_load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.model_loaded = False
            self._set_status("模型文件未找到 — 请先运行 train_model.py 生成模型", DANGER)
            self._set_pill("模型缺失", DANGER)
            self.result_label.configure(text="未就绪", text_color=DANGER)
            self.conf_label.configure(text="请补齐 hydrogen_model.pkl 后重启")
            return
        try:
            bundle = joblib.load(MODEL_PATH)
            self.model = bundle["model"]
            self.feature_names = bundle["feature_names"]
            self.model.n_jobs = 1
            self.model_loaded = True
            self._set_status("模型已加载，等待输入参数…", OK)
            self._set_pill("模型就绪", OK)
        except Exception as e:
            self.model_loaded = False
            self._set_status(f"模型加载失败：{e}", DANGER)
            self._set_pill("加载失败", DANGER)

    # ── 判识 ─────────────────────────────────────────────
    def _reset_inputs(self):
        for i, row in enumerate(self.sliders):
            row.reset(SLIDER_DEFS[i][3])
        self.tectonic_var.set(TECTONIC_OPTIONS[0])
        self._set_status("参数已恢复默认值", MUTED)

    def _on_predict(self):
        if not self.model_loaded:
            self._try_load_model()
            if not self.model_loaded:
                messagebox.showwarning("模型未就绪", "请先运行 train_model.py 训练模型。")
                return

        raw = {
            "delta_D": float(self.sliders[0].get()),
            "ln_CH4_H2": float(self.sliders[1].get()),
            "R_Ra": float(self.sliders[2].get()),
            "C1_C1_plus": float(self.sliders[3].get()),
            "delta_13C_CH4": float(self.sliders[4].get()),
            "Tectonic_Setting": self.tectonic_var.get(),
        }
        frame = pd.DataFrame([raw])
        frame = pd.get_dummies(frame, columns=["Tectonic_Setting"])
        frame = frame.reindex(columns=self.feature_names, fill_value=0)

        pred = self.model.predict(frame)[0]
        probs = dict(zip(self.model.classes_, self.model.predict_proba(frame)[0]))

        try:
            log_prediction(raw, str(pred))
            saved = True
        except Exception as e:
            saved = False
            self._set_status(f"判识完成，但记录写入失败：{e}", DANGER)

        color = CLASS_COLORS.get(pred, ACCENT)
        self.result_label.configure(text=pred, text_color=color)
        self.conf_label.configure(
            text=f"综合置信度 {probs[pred] * 100:.1f}%　·　输入参数已归档")
        for name, row in self.prob_rows.items():
            row.set(float(probs.get(name, 0.0)))

        if saved:
            self._set_status(f"判识完成 — 最可能成因：{pred}（已写入审计日志）", OK)
            if self._current == TABS[2]:
                self._refresh_history()

    # ── 题库 ─────────────────────────────────────────────
    def _load_training_data(self):
        if self._training_loaded:
            return
        try:
            if not os.path.exists(DATASET_PATH):
                self.bank_desc.configure(
                    text=f"⚠ 未找到 {os.path.basename(DATASET_PATH)}，请先运行 build_dataset.py 生成数据集。",
                    text_color=DANGER)
                return
            df = pd.read_csv(DATASET_PATH)
            self.bank_table.clear()
            for i, row in enumerate(df.itertuples(index=False)):
                self.bank_table.insert_row([
                    f"{row.delta_D:.1f}",
                    f"{row.ln_CH4_H2:.3f}",
                    f"{row.R_Ra:.4f}",
                    f"{row.C1_C1_plus:.4f}",
                    f"{row.delta_13C_CH4:.1f}",
                    row.Tectonic_Setting,
                    row.Origin,
                ], i)
            self._training_loaded = True
            self.bank_desc.configure(
                text=f"共 {len(df)} 条样本，四类成因均匀分布并含参数重叠区；仅用于说明模型训练依据，不作为地质证据。",
                text_color=MUTED)
        except Exception as e:
            self.bank_desc.configure(text=f"⚠ 加载训练数据失败：{e}", text_color=DANGER)

    # ── 日志 ─────────────────────────────────────────────
    def _refresh_history(self):
        try:
            df = get_prediction_history(limit=200)
            self.log_table.clear()
            if df.empty:
                self.log_desc.configure(text="暂无记录，完成一次判识后自动写入", text_color=MUTED)
                return
            for i, row in enumerate(df.itertuples(index=False)):
                self.log_table.insert_row([
                    str(row.id),
                    str(row.timestamp),
                    f"{row.delta_D:.1f}",
                    f"{row.ln_CH4_H2:.3f}",
                    f"{row.R_Ra:.4f}",
                    f"{row.C1_C1_plus:.4f}",
                    f"{row.delta_13C_CH4:.1f}",
                    row.Tectonic_Setting,
                    row.predicted_origin,
                ], i)
            self.log_desc.configure(
                text=f"共 {len(df)} 条记录（最多显示最近 200 条），双击行可查看明细",
                text_color=MUTED)
        except Exception as e:
            self.log_desc.configure(text=f"⚠ 历史记录加载失败：{e}", text_color=DANGER)

    def _on_row_detail(self, _event):
        tree = self.log_table.tree
        selection = tree.selection()
        if not selection:
            return
        values = tree.item(selection[0], "values")
        if not values:
            return

        dlg = ctk.CTkToplevel(self)
        dlg.title("判识详情")
        dlg.geometry("470x500")
        x = self.winfo_rootx() + (self.winfo_width() - 470) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 500) // 3
        dlg.geometry(f"470x500+{x}+{y}")
        dlg.configure(fg_color=BG)
        dlg.attributes("-topmost", True)
        dlg.after(120, lambda: dlg.lift())

        ctk.CTkLabel(dlg, text="判识详情", font=f(17, True), text_color=TEXT).pack(pady=(20, 2))
        ctk.CTkLabel(dlg, text=f"记录编号 {values[0]}", font=f(10),
                     text_color=MUTED).pack(pady=(0, 14))

        card = Card(dlg)
        card.pack(fill="x", padx=22, pady=(0, 16))
        card.grid_columnconfigure(1, weight=1)

        fields = [
            ("判识时间", values[1]),
            ("δD (‰)", values[2]),
            ("ln(CH₄/H₂)", values[3]),
            ("R/Ra", values[4]),
            ("C₁/C₁₊", values[5]),
            ("δ¹³C-CH₄ (‰)", values[6]),
            ("构造背景", values[7]),
        ]
        for i, (label, value) in enumerate(fields):
            ctk.CTkLabel(card, text=label, font=f(10), text_color=MUTED,
                         width=112, anchor="e").grid(row=i, column=0, padx=(18, 10), pady=5,
                                                     sticky="e")
            ctk.CTkLabel(card, text=value, font=("Consolas", 11), text_color=TEXT,
                         anchor="w").grid(row=i, column=1, padx=(0, 18), pady=5, sticky="w")

        ctk.CTkLabel(card, text="AI 结论", font=f(10), text_color=MUTED,
                     width=112, anchor="e").grid(row=len(fields), column=0, padx=(18, 10),
                                                 pady=(12, 8), sticky="e")
        ctk.CTkLabel(card, text=values[8], font=f(13, True),
                     text_color=CLASS_COLORS.get(values[8], ACCENT),
                     anchor="w").grid(row=len(fields), column=1, padx=(0, 18),
                                      pady=(12, 8), sticky="w")

        ctk.CTkButton(dlg, text="关闭", command=dlg.destroy, font=f(11), height=34,
                      corner_radius=8, fg_color=CARD_2, hover_color="#2A3A44",
                      text_color=TEXT, width=110).pack(pady=(0, 18))


if __name__ == "__main__":
    app = App()
    app.mainloop()
