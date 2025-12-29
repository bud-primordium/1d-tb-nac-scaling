"""统一绘图接口与样式设置。"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _pick_cjk_fonts() -> list[str]:
    """从系统字体中挑选可用的中文字体。"""
    candidates = [
        "PingFang SC",
        "Heiti SC",
        "Heiti TC",
        "Songti SC",
        "Songti TC",
        "STHeiti",
        "STHeiti Medium",
        "STHeiti Light",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "WenQuanYi Micro Hei",
        "SimHei",
        "STSong",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    return [name for name in candidates if name in available]


def setup_plot_style() -> None:
    """设置中文字体与统一风格。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fonts = _pick_cjk_fonts()
    if not fonts:
        fonts = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def plot_scaling(
    ax: plt.Axes,
    n_vals: np.ndarray,
    y_vals: np.ndarray,
    label: str,
    color: str,
    fit_min_n: int = 40,
) -> Tuple[float, float]:
    """绘制标度律并拟合斜率。"""
    mask = n_vals >= fit_min_n
    log_n = np.log(n_vals[mask])
    log_y = np.log(y_vals[mask])
    slope, intercept = np.polyfit(log_n, log_y, 1)
    ax.scatter(n_vals, y_vals, label=label, color=color)
    fit_y = np.exp(intercept) * n_vals ** slope
    ax.plot(n_vals, fit_y, linestyle="--", color=color, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\langle d^2 \rangle$")
    ax.legend()
    ax.set_title(f"{label}，拟合斜率 β = {-slope:.2f}")
    return float(slope), float(intercept)


def save_figure(fig: plt.Figure, path_base: str, dpi: int = 150) -> None:
    """同时保存 PDF 和 PNG。"""
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_base}.png", dpi=dpi, bbox_inches="tight")
