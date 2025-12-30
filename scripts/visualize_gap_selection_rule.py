"""带隙选择定则可视化：验证 q≠0 的声子不影响直接带隙的一阶响应。

Figure 8: 带隙一阶响应系数 |g_cc - g_vv|² vs q
Figure 9: 冻结声子奇偶性验证（可选）

采用 publication_figures 风格，输出到 results/figures/ 目录。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import matplotlib.pyplot as plt

from src.diagnostics import select_k_from_ratio
from src.gap_selection_rule import compute_gap_response
from src.lattice import q_grid
from src.phonon_diatomic import displacement_from_q, diatomic_modes_grid
from src.ssh_electron import build_hamiltonian, diagonalize


# =============================================================================
# 样式设置 (与 publication_figures 一致)
# =============================================================================

COLORS = {
    'case1': '#1f77b4',  # 蓝色
    'case2': '#ff7f0e',  # 橙色
    'case3': '#2ca02c',  # 绿色
    'theory': '#7f7f7f',  # 灰色
    'highlight': '#d62728',  # 红色
}


def setup_publication_style():
    """设置出版级图表样式。"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'PingFang SC', 'Heiti SC'],
        'font.size': 10,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'axes.labelpad': 4,
        'axes.unicode_minus': False,
        'axes.formatter.useoffset': False,
        'axes.formatter.use_mathtext': True,
        'xtick.major.size': 4,
        'xtick.major.width': 1.0,
        'xtick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.major.size': 4,
        'ytick.major.width': 1.0,
        'ytick.labelsize': 10,
        'ytick.direction': 'in',
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.grid': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def ensure_output_dir():
    """确保输出目录存在。"""
    output_dir = ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fig8_gap_response_vs_q(
    output_dir: Path,
    n_cells: int,
    r_k: float,
    t0: float,
    delta_t: float,
    alpha: float,
    k_spring: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
) -> None:
    """Figure 8: 带隙一阶响应系数 |g_cc - g_vv|² vs q。"""
    # 选择 k 点
    k, _ = select_k_from_ratio(n_cells, r_k, a=a)

    # q 网格
    q_vals = q_grid(n_cells, a=a)

    # 计算带隙响应
    result = compute_gap_response(
        n_cells=n_cells,
        k=k,
        q_vals=q_vals,
        t0=t0,
        delta_t=delta_t,
        alpha=alpha,
        k_spring=k_spring,
        a=a,
        mass_a=mass_a,
        mass_b=mass_b,
    )

    # 对两个声子支求和
    gap_response_sum = np.sum(result.gap_response, axis=1)

    # 找到 q=0 的位置
    q0_idx = np.argmin(np.abs(q_vals))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # 线性坐标
    ax = axes[0]
    ax.stem(q_vals / np.pi * a, gap_response_sum, basefmt=" ", linefmt=COLORS['case1'], markerfmt="o")
    ax.axvline(0, linestyle="--", color=COLORS['highlight'], alpha=0.5, label="$q=0$")
    ax.set_xlabel(r"$q$ ($\pi/a$)")
    ax.set_ylabel(r"$|g_{cc} - g_{vv}|^2$ (sum over modes)")
    ax.set_title(f"Gap first-order response vs $q$ ($N={n_cells}$, $k={k/np.pi:.2f}\\pi$)", fontsize=11)
    ax.legend(fontsize=9)

    # 标注 q=0 的值
    q0_val = gap_response_sum[q0_idx]
    ax.annotate(
        f"$q=0$: {q0_val:.2e}",
        xy=(0, q0_val),
        xytext=(0.3, q0_val * 0.8),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=COLORS['highlight']),
    )

    # 对数坐标（更清晰地展示数值零）
    ax = axes[1]
    # 将数值零替换为机器精度以便在对数坐标上显示
    gap_response_log = np.where(gap_response_sum > 1e-30, gap_response_sum, 1e-30)
    ax.semilogy(q_vals / np.pi * a, gap_response_log, "o-", color=COLORS['case1'], markersize=6)
    ax.axvline(0, linestyle="--", color=COLORS['highlight'], alpha=0.5)
    ax.axhline(1e-20, linestyle=":", color=COLORS['theory'], alpha=0.5, label=r"$10^{-20}$ threshold")
    ax.set_xlabel(r"$q$ ($\pi/a$)")
    ax.set_ylabel(r"$|g_{cc} - g_{vv}|^2$ (log scale)")
    ax.set_title("Log scale: should be numerical zero for $q \\neq 0$", fontsize=11)
    ax.legend(fontsize=9)

    # 统计 q≠0 的最大值
    non_q0_vals = np.delete(gap_response_sum, q0_idx)
    max_non_q0 = np.max(non_q0_vals) if non_q0_vals.size > 0 else 0
    ax.text(
        0.97, 0.95,
        f"max($q \\neq 0$) = {max_non_q0:.2e}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(output_dir / "fig8_gap_response_vs_q.png")
    plt.close(fig)

    # 打印验证信息
    print(f"    q=0 response: {q0_val:.4e}")
    print(f"    max(q≠0) response: {max_non_q0:.4e}")
    if max_non_q0 < 1e-20:
        print("    ✓ Verified: q≠0 response is numerical zero")
    else:
        print("    ⚠ Warning: q≠0 response is not zero")
    print("  Figure 8: Gap response vs q - done")


def fig9_frozen_phonon_parity(
    output_dir: Path,
    n_cells: int,
    r_k: float,
    t0: float,
    delta_t: float,
    k_spring: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
    q_test_ratio: float = 0.2,
    amplitude_max: float = 0.1,
    n_points: int = 21,
) -> None:
    """Figure 9: 冻结声子奇偶性验证。"""
    # 选择测试的 q 点（q≠0）
    q_test, _ = select_k_from_ratio(n_cells, q_test_ratio, a=a)

    # 获取该 q 的声子模式
    q_vals_single = np.array([q_test])
    omegas, evecs = diatomic_modes_grid(q_vals_single, k_spring, mass_a, mass_b, a=a)

    # 选择光学支（通常频率更高）
    mode_idx = int(np.argmax(omegas[0]))
    evec = evecs[0, :, mode_idx]

    # 振幅扫描
    amplitudes = np.linspace(-amplitude_max, amplitude_max, n_points)
    gaps = []

    for amp in amplitudes:
        # 构造位移场
        q_amp = np.array([[0.0, amp]])  # 只激发光学模式
        evecs_for_disp = evecs.copy()
        evecs_for_disp[0, :, 0] = 0  # 关闭声学模式
        evecs_for_disp[0, :, 1] = evec  # 光学模式

        disp = displacement_from_q(
            q_vals_single,
            q_amp,
            evecs_for_disp,
            n_cells,
            a=a,
            mass_a=mass_a,
            mass_b=mass_b,
        )

        # 构造含位移的哈密顿量并对角化
        h = build_hamiltonian(
            n_cells,
            t0=t0,
            delta_t=delta_t,
            displacements=disp,
            alpha=0.5,
            pbc=True,
        )
        evals, _ = diagonalize(h)

        # 计算带隙
        n_states = len(evals)
        gap = evals[n_states // 2] - evals[n_states // 2 - 1]
        gaps.append(gap)

    gaps = np.array(gaps)
    gap_0 = gaps[n_points // 2]  # A=0 时的带隙

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 带隙 vs 振幅
    ax = axes[0]
    ax.plot(amplitudes, gaps, "o-", color=COLORS['case1'], markersize=6)
    ax.axvline(0, linestyle="--", color=COLORS['theory'], alpha=0.5)
    ax.axhline(gap_0, linestyle=":", color=COLORS['highlight'], alpha=0.5, label=f"$E_g(0) = {gap_0:.4f}$")
    ax.set_xlabel("Displacement amplitude $A$")
    ax.set_ylabel(r"Band gap $E_g$")
    ax.set_title(f"Frozen phonon: $E_g$ vs $A$ ($q={q_test/np.pi:.2f}\\pi$)", fontsize=11)
    ax.legend(fontsize=9)

    # 带隙变化 vs 振幅（检验奇偶性）
    ax = axes[1]
    delta_gap = gaps - gap_0
    ax.plot(amplitudes, delta_gap, "o-", color=COLORS['case3'], markersize=6)
    ax.axvline(0, linestyle="--", color=COLORS['theory'], alpha=0.5)
    ax.axhline(0, linestyle=":", color=COLORS['highlight'], alpha=0.5)
    ax.set_xlabel("Displacement amplitude $A$")
    ax.set_ylabel(r"$\Delta E_g = E_g(A) - E_g(0)$")
    ax.set_title("Gap change: expected even function (quadratic)", fontsize=11)

    # 检验奇偶性
    idx_plus = n_points // 2 + n_points // 4
    idx_minus = n_points // 2 - n_points // 4
    delta_plus = delta_gap[idx_plus]
    delta_minus = delta_gap[idx_minus]
    asymmetry = abs(delta_plus - delta_minus) / (abs(delta_plus) + abs(delta_minus) + 1e-30)

    ax.text(
        0.97, 0.05,
        f"Symmetry check:\n$\\Delta E_g(+A) = {delta_plus:.4e}$\n$\\Delta E_g(-A) = {delta_minus:.4e}$\nAsymmetry = {asymmetry:.2e}",
        transform=ax.transAxes,
        fontsize=8,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(output_dir / "fig9_frozen_phonon_parity.png")
    plt.close(fig)

    print(f"    q = {q_test/np.pi:.2f}π frozen phonon test:")
    print(f"    ΔE_g(+A) = {delta_plus:.4e}")
    print(f"    ΔE_g(-A) = {delta_minus:.4e}")
    print(f"    Asymmetry = {asymmetry:.2e}")
    if asymmetry < 0.1:
        print("    ✓ Verified: gap change is even function (first-order term is zero)")
    else:
        print("    ⚠ Warning: odd component exists")
    print("  Figure 9: Frozen phonon parity - done")


def main() -> None:
    """执行全部带隙选择定则可视化。"""
    setup_publication_style()
    output_dir = ensure_output_dir()

    # 参数
    n_cells = 40
    r_k = 0.1
    t0 = 1.0
    delta_t = 0.2
    alpha = 0.5
    k_spring = 1.0
    a = 1.0
    mass_a = 1.0
    mass_b = 1.0

    print(f"Output directory: {output_dir}\n")

    print("Generating Figure 8: Gap response vs q...")
    fig8_gap_response_vs_q(
        output_dir=output_dir,
        n_cells=n_cells,
        r_k=r_k,
        t0=t0,
        delta_t=delta_t,
        alpha=alpha,
        k_spring=k_spring,
        a=a,
        mass_a=mass_a,
        mass_b=mass_b,
    )

    print("Generating Figure 9: Frozen phonon parity...")
    fig9_frozen_phonon_parity(
        output_dir=output_dir,
        n_cells=n_cells,
        r_k=r_k,
        t0=t0,
        delta_t=delta_t,
        k_spring=k_spring,
        a=a,
        mass_a=mass_a,
        mass_b=mass_b,
        q_test_ratio=0.2,
        amplitude_max=0.1,
        n_points=21,
    )

    print("\nGap selection rule visualization complete!")


if __name__ == "__main__":
    main()
