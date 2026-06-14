import customtkinter as ctk
import pandas as pd
import numpy as np
import joblib
import os
import tkinter as tk
from tkinter import ttk, messagebox

from db_manager import init_db, log_prediction, get_prediction_history

# ── Paths ──────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "hydrogen_model.pkl")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "hydrogen_realistic_dataset.csv")

# ── Slider definitions ─────────────────────────────────
SLIDER_DEFS = [
    ("δD (‰)",                -850.0, -100.0, -500.0, ".0f"),
    ("ln(CH₄/H₂)",            -15.0,   15.0,   -5.0, ".1f"),
    ("R/Ra",                    0.01,   8.0,    1.0, ".2f"),
    ("C₁/C₁₊",                 0.80,   1.0,   0.95, ".3f"),
    ("δ¹³C-CH₄ (‰)",         -80.0,    0.0,  -40.0, ".1f"),
]

TECTONIC_OPTIONS = ["活动断裂带", "超基性岩/洋壳", "沉积盆地"]


# ═══════════════════════════════════════════════════════════════
# Helper: slider + label group (unchanged)
# ═══════════════════════════════════════════════════════════════
class SliderGroup(ctk.CTkFrame):
    def __init__(self, parent, label, from_val, to_val, default_val, fmt):
        super().__init__(parent, fg_color="transparent")
        self.fmt = fmt

        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(fill="x", pady=(6, 0))

        ctk.CTkLabel(top_row, text=label,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")

        self.value_label = ctk.CTkLabel(top_row, text=f"{default_val:{fmt}}",
                                        font=ctk.CTkFont(size=13),
                                        text_color="#5bc0de")
        self.value_label.pack(side="right")

        self.slider = ctk.CTkSlider(
            self, from_=from_val, to=to_val,
            command=self._on_change,
            height=18,
            progress_color="#5bc0de",
            button_color="#5bc0de",
            button_hover_color="#3a9eb5",
        )
        self.slider.set(default_val)
        self.slider.pack(fill="x", pady=(2, 4))

    def _on_change(self, val):
        self.value_label.configure(text=f"{val:{self.fmt}}")

    def get(self) -> float:
        return self.slider.get()


# ═══════════════════════════════════════════════════════════════
# Helper: dark-themed ttk.Treeview wrapped in CTkFrame
# ═══════════════════════════════════════════════════════════════
class DarkTreeview(ctk.CTkFrame):
    """A ttk.Treeview with dark theme + scrollbars, inside a CTkFrame."""

    _style_initialized = False

    def __init__(self, parent, columns, headings, widths,
                 anchor="center", row_height=28):
        super().__init__(parent, fg_color="transparent")

        self._init_style(row_height)

        # Scrollbars
        self.v_scroll = ctk.CTkScrollbar(self, orientation="vertical")
        self.h_scroll = ctk.CTkScrollbar(self, orientation="horizontal")

        # Treeview
        self.tree = ttk.Treeview(
            self, columns=columns, show="headings",
            style="Dark.Treeview",
            yscrollcommand=self._on_vscroll,
            xscrollcommand=self._on_hscroll,
        )

        # Layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.v_scroll.configure(command=self.tree.yview)
        self.h_scroll.configure(command=self.tree.xview)

        # Column config
        for col, heading, width in zip(columns, headings, widths):
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, minwidth=40, anchor=anchor)

    @classmethod
    def _init_style(cls, row_height):
        if cls._style_initialized:
            return
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background="#2b2b2b",
                        foreground="#ffffff",
                        fieldbackground="#2b2b2b",
                        borderwidth=0,
                        rowheight=row_height)
        style.configure("Dark.Treeview.Heading",
                        background="#3b3b3b",
                        foreground="#dddddd",
                        borderwidth=1,
                        relief="flat",
                        font=("", 10, "bold"))
        style.map("Dark.Treeview",
                  background=[("selected", "#5bc0de")],
                  foreground=[("selected", "#1a1a1a")])
        style.map("Dark.Treeview.Heading",
                  background=[("active", "#4a4a4a")])
        cls._style_initialized = True

    def _on_vscroll(self, *args):
        self.v_scroll.set(*args)
        self.tree.yview(*args)

    def _on_hscroll(self, *args):
        self.h_scroll.set(*args)
        self.tree.xview(*args)

    def clear(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def insert_row(self, values, tag=""):
        return self.tree.insert("", "end", values=values, tags=(tag,) if tag else ())


# ═══════════════════════════════════════════════════════════════
# Main Application — 3-tab architecture
# ═══════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window config
        self.title("氢气成因多维智能判识系统")
        self.geometry("1080x740")
        self.minsize(900, 620)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Model state
        self.model = None
        self.feature_names = None
        self.model_loaded = False

        # Lazy-load flags
        self._training_data_loaded = False
        self._last_tab = ""

        # Init database
        try:
            init_db()
        except Exception as e:
            messagebox.showwarning("数据库错误", f"无法初始化数据库:\n{e}")

        # Build
        self._build_ui()
        self._try_load_model()
        self._start_tab_monitor()

    # ═══════════════════════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════════════════════
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        # ── Header ──
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=(18, 0), sticky="ew")
        ctk.CTkLabel(header, text="氢气成因多维智能判识系统",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, text="v2.0  ·  Hydrogen Genesis Intelligent Identifier",
                     font=ctk.CTkFont(size=11), text_color="#888").pack(side="left", padx=(12, 0))

        # ── Top-Level TabView ──
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, padx=18, pady=(10, 0), sticky="nsew")

        self.tabview.add("AI 智能判识中心")
        self.tabview.add("模型标准训练题库")
        self.tabview.add("历史预测审计日志")

        self._build_tab1(self.tabview.tab("AI 智能判识中心"))
        self._build_tab2(self.tabview.tab("模型标准训练题库"))
        self._build_tab3(self.tabview.tab("历史预测审计日志"))

        # ── Status bar ──
        self.status_bar = ctk.CTkLabel(
            self, text="",
            font=ctk.CTkFont(size=11), text_color="#888", anchor="w",
        )
        self.status_bar.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

    # ═══════════════════════════════════════════════════════════
    # Tab 1: AI 智能判识中心
    # ═══════════════════════════════════════════════════════════
    def _build_tab1(self, parent):
        parent.grid_columnconfigure(0, weight=0)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        # Left: input panel
        left = ctk.CTkFrame(parent)
        left.grid(row=0, column=0, padx=(8, 6), pady=10, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="■  地质参数输入",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     anchor="w").pack(fill="x", padx=14, pady=(14, 10))

        # Tectonic dropdown
        dd_frame = ctk.CTkFrame(left, fg_color="transparent")
        dd_frame.pack(fill="x", padx=14, pady=(0, 4))
        ctk.CTkLabel(dd_frame, text="构造背景",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.tectonic_var = ctk.StringVar(value=TECTONIC_OPTIONS[0])
        self.tectonic_dd = ctk.CTkOptionMenu(
            dd_frame, values=TECTONIC_OPTIONS,
            variable=self.tectonic_var,
            font=ctk.CTkFont(size=12),
            dropdown_font=ctk.CTkFont(size=12),
            fg_color="#3b3b3b",
            button_color="#5bc0de",
            button_hover_color="#3a9eb5",
        )
        self.tectonic_dd.pack(fill="x", pady=(2, 6))

        # Sliders
        self.sliders: list[SliderGroup] = []
        for label, frm, to, dft, fmt in SLIDER_DEFS:
            sg = SliderGroup(left, label, frm, to, dft, fmt)
            sg.pack(fill="x", padx=14)
            self.sliders.append(sg)

        # Right: output panel
        right = ctk.CTkFrame(parent)
        right.grid(row=0, column=1, padx=(6, 8), pady=10, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="■  AI 判识结果",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     anchor="w").pack(fill="x", padx=14, pady=(14, 10))

        # Predict button
        self.btn_predict = ctk.CTkButton(
            right, text="启动 AI 判识",
            command=self._on_predict,
            font=ctk.CTkFont(size=17, weight="bold"),
            height=50,
            corner_radius=10,
            fg_color="#5bc0de",
            hover_color="#3a9eb5",
            text_color="#1a1a1a",
        )
        self.btn_predict.pack(fill="x", padx=14, pady=(0, 12))

        # Result
        self.result_header = ctk.CTkLabel(right, text="最可能成因",
                                          font=ctk.CTkFont(size=12),
                                          text_color="#888")
        self.result_header.pack(anchor="w", padx=14)
        self.result_label = ctk.CTkLabel(right, text="—",
                                         font=ctk.CTkFont(size=27, weight="bold"),
                                         text_color="#5bc0de")
        self.result_label.pack(anchor="w", padx=14, pady=(2, 8))

        # Probability textbox
        self.prob_header = ctk.CTkLabel(right, text="各成因概率分布",
                                        font=ctk.CTkFont(size=12),
                                        text_color="#888")
        self.prob_header.pack(anchor="w", padx=14)
        self.prob_text = ctk.CTkTextbox(
            right,
            font=ctk.CTkFont(size=13),
            fg_color="#2b2b2b",
            border_width=1, border_color="#444",
            corner_radius=8,
            activate_scrollbars=False,
        )
        self.prob_text.pack(fill="both", expand=True, padx=14, pady=(2, 14))
        self.prob_text.configure(state="disabled")

    # ═══════════════════════════════════════════════════════════
    # Tab 2: 模型标准训练题库
    # ═══════════════════════════════════════════════════════════
    def _build_tab2(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=1)

        # Description
        self.lbl_training_desc = ctk.CTkLabel(
            parent, text="",
            font=ctk.CTkFont(size=12),
            text_color="#aaa",
            anchor="w", justify="left",
        )
        self.lbl_training_desc.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        # Table
        train_cols = ["delta_D", "ln_CH4_H2", "R_Ra", "C1_C1_plus", "delta_13C_CH4",
                      "Tectonic_Setting", "Origin"]
        train_heads = ["δD (‰)", "ln(CH₄/H₂)", "R/Ra", "C₁/C₁₊", "δ¹³C-CH₄ (‰)",
                       "构造背景", "成因类型"]
        train_widths = [90, 100, 80, 90, 110, 120, 120]

        self.train_table = DarkTreeview(parent, train_cols, train_heads, train_widths,
                                        anchor="center")
        self.train_table.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _load_training_data(self):
        """Lazy-load training data from CSV into Tab 2's treeview."""
        if self._training_data_loaded:
            return
        try:
            if not os.path.exists(DATASET_PATH):
                self.lbl_training_desc.configure(
                    text="⚠ 未找到 hydrogen_training_dataset.csv，请先运行 build_dataset.py 生成数据集。")
                return
            df = pd.read_csv(DATASET_PATH)
            self.train_table.clear()
            for _, row in df.iterrows():
                self.train_table.insert_row([
                    f"{row['delta_D']:.1f}",
                    f"{row['ln_CH4_H2']:.3f}",
                    f"{row['R_Ra']:.4f}",
                    f"{row['C1_C1_plus']:.4f}",
                    f"{row['delta_13C_CH4']:.1f}",
                    row["Tectonic_Setting"],
                    row["Origin"],
                ])
            self.lbl_training_desc.configure(
                text=f"本数据集由系统根据《地质体中天然氢气成因识别方法》"
                     f"等前沿文献地化边界自动合成，共计 {len(df)} 条，"
                     f"为底层随机森林模型提供先验训练支持。"
            )
            self._training_data_loaded = True
        except Exception as e:
            self.lbl_training_desc.configure(text=f"⚠ 加载训练数据失败: {e}")

    # ═══════════════════════════════════════════════════════════
    # Tab 3: 历史预测审计日志
    # ═══════════════════════════════════════════════════════════
    def _build_tab3(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ctk.CTkFrame(parent, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        ctk.CTkLabel(toolbar, text="所有用户预测记录（按时间倒序排列 · 双击行查看详情）",
                     font=ctk.CTkFont(size=12), text_color="#888").pack(side="left")
        ctk.CTkButton(toolbar, text="刷新",
                      command=self._refresh_history,
                      font=ctk.CTkFont(size=12),
                      fg_color="#555", hover_color="#777",
                      width=60, height=28).pack(side="right")

        # Table
        hist_cols = ["id", "timestamp", "delta_D", "ln_CH4_H2", "R_Ra",
                     "C1_C1_plus", "delta_13C_CH4", "Tectonic_Setting", "predicted_origin"]
        hist_heads = ["#", "判识时间", "δD (‰)", "ln(CH₄/H₂)", "R/Ra",
                      "C₁/C₁₊", "δ¹³C-CH₄ (‰)", "构造背景", "AI 最终结论"]
        hist_widths = [45, 150, 80, 90, 70, 80, 100, 110, 110]

        self.hist_table = DarkTreeview(parent, hist_cols, hist_heads, hist_widths,
                                       anchor="center", row_height=26)
        self.hist_table.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        # Double-click → detail popup
        self.hist_table.tree.bind("<Double-1>", self._on_history_double_click)

    def _refresh_history(self):
        try:
            df = get_prediction_history(limit=200)
            self.hist_table.clear()
            if df.empty:
                return
            for _, row in df.iterrows():
                self.hist_table.insert_row([
                    str(row["id"]),
                    str(row["timestamp"]),
                    f"{row['delta_D']:.1f}",
                    f"{row['ln_CH4_H2']:.3f}",
                    f"{row['R_Ra']:.4f}",
                    f"{row['C1_C1_plus']:.4f}",
                    f"{row['delta_13C_CH4']:.1f}",
                    row["Tectonic_Setting"],
                    row["predicted_origin"],
                ])
        except Exception as e:
            self._set_status(f"⚠ 历史记录加载失败: {e}", "#e07b4b")

    def _on_history_double_click(self, event):
        """双击历史表格行 → 弹出详情窗口。"""
        tree = self.hist_table.tree
        sel = tree.selection()
        if not sel:
            return
        values = tree.item(sel[0], "values")
        if not values:
            return

        detail = ctk.CTkToplevel(self)
        detail.title("预测审计详情")
        detail.geometry("420x380")
        detail.after(200, lambda: detail.lift())

        ctk.CTkLabel(detail, text="预测审计详情",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(16, 8))

        info = ctk.CTkFrame(detail, fg_color="transparent")
        info.pack(fill="both", expand=True, padx=20)

        fields = [
            ("记录编号",   values[0]),
            ("判识时间",   values[1]),
            ("δD (‰)",              values[2]),
            ("ln(CH₄/H₂)",          values[3]),
            ("R/Ra",                values[4]),
            ("C₁/C₁₊",              values[5]),
            ("δ¹³C-CH₄ (‰)",        values[6]),
            ("构造背景",              values[7]),
            ("AI 最终结论",           values[8]),
        ]

        for label, value in fields:
            rf = ctk.CTkFrame(info, fg_color="transparent")
            rf.pack(fill="x", pady=2)
            ctk.CTkLabel(rf, text=f"{label}:", width=105, anchor="e",
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color="#888").pack(side="left", padx=(0, 8))
            color = "#5bc0de" if label == "AI 最终结论" else "#ddd"
            ctk.CTkLabel(rf, text=value,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=color).pack(side="left")

        ctk.CTkButton(detail, text="关闭", command=detail.destroy,
                      fg_color="#555", hover_color="#777",
                      font=ctk.CTkFont(size=12)).pack(pady=(0, 14))

    # ═══════════════════════════════════════════════════════════
    # Tab change monitor (polls every 500ms)
    # ═══════════════════════════════════════════════════════════
    def _start_tab_monitor(self):
        self._last_tab = self.tabview.get()
        self._check_tab_change()

    def _check_tab_change(self):
        current = self.tabview.get()
        if current != self._last_tab:
            self._last_tab = current
            if current == "模型标准训练题库":
                self._load_training_data()
            elif current == "历史预测审计日志":
                self._refresh_history()
        self.after(500, self._check_tab_change)

    # ═══════════════════════════════════════════════════════════
    # Model loading
    # ═══════════════════════════════════════════════════════════
    def _try_load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.model_loaded = False
            self._set_status("⚠ 模型文件未找到 — 请先运行 train_model.py 生成模型", "#e07b4b")
            self.result_label.configure(text="未就绪", text_color="#e07b4b")
            return

        try:
            bundle = joblib.load(MODEL_PATH)
            self.model = bundle["model"]
            self.feature_names = bundle["feature_names"]
            self.model_loaded = True
            self._set_status("● 模型已加载，等待输入参数...", "#5cb85c")
        except Exception as e:
            self.model_loaded = False
            self._set_status(f"⚠ 模型加载失败: {e}", "#e07b4b")

    # ═══════════════════════════════════════════════════════════
    # Prediction logic
    # ═══════════════════════════════════════════════════════════
    def _on_predict(self):
        if not self.model_loaded:
            self._try_load_model()
            if not self.model_loaded:
                messagebox.showwarning("模型未就绪", "请先运行 train_model.py 训练模型。")
                return

        tectonic = self.tectonic_var.get()
        values = [sg.get() for sg in self.sliders]

        raw = {
            "delta_D":        values[0],
            "ln_CH4_H2":      values[1],
            "R_Ra":           values[2],
            "C1_C1_plus":     values[3],
            "delta_13C_CH4":  values[4],
            "Tectonic_Setting": tectonic,
        }
        input_df = pd.DataFrame([raw])
        input_encoded = pd.get_dummies(input_df, columns=["Tectonic_Setting"])
        input_encoded = input_encoded.reindex(columns=self.feature_names, fill_value=0)

        pred = self.model.predict(input_encoded)[0]
        probs = self.model.predict_proba(input_encoded)[0]

        # Log to DB
        try:
            log_prediction(raw, str(pred))
        except Exception as e:
            self._set_status(f"⚠ 预测记录写入失败: {e}", "#e07b4b")

        # Update result display
        self.result_label.configure(text=pred, text_color="#5bc0de")

        prob_pairs = sorted(
            zip(self.model.classes_, probs),
            key=lambda x: x[1], reverse=True,
        )
        prob_lines = []
        for cls, p in prob_pairs:
            bar = "█" * int(p * 30)
            prob_lines.append(f"  {cls:<12s}  {bar}  {p * 100:6.2f}%")

        self.prob_text.configure(state="normal")
        self.prob_text.delete("1.0", "end")
        self.prob_text.insert("1.0", "\n".join(prob_lines))
        self.prob_text.configure(state="disabled")

        self._set_status(f"● 判识完成 — 最可能成因: {pred}（已存入数据库）", "#5cb85c")

    # ═══════════════════════════════════════════════════════════
    # Status bar
    # ═══════════════════════════════════════════════════════════
    def _set_status(self, text, color="#888"):
        self.status_bar.configure(text=text, text_color=color)


# ── Entry Point ─────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
