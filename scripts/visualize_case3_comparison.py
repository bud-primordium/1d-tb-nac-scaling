"""Case 3 严格版 vs 近似版对比可视化。

Figure 1: 标度律对比 (Case 3 vs Case 3')
Figure 2: 前因子相对差异 vs N
Figure 3: Bloch 特征收敛 (W_±k 指标)
Figure 4/5: 实空间/倒空间波函数对比

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

from src.coupling_density import kspace_distribution
from src.diagnostics import ipr, select_k_from_ratio
from src.experiments_simple import (
    bloch_character_vs_N,
    case3_scaling,
    case3_scaling_strict,
)
from src.scattering_state import select_scattering_state
from src.tb_electron_1band import (
    bloch_state,
    build_hamiltonian,
    build_square_well,
    diagonalize,
    dispersion,
)


# =============================================================================
# 样式设置 (与 publication_figures 一致)
# =============================================================================

COLORS = {
    'case1': '#1f77b4',  # 蓝色
    'case2': '#ff7f0e',  # 橙色
    'case3': '#2ca02c',  # 绿色
    'theory': '#7f7f7f',  # 灰色
    'highlight': '#d62728',  # 红色
    'approx': '#1f77b4',  # 近似版用蓝色
    'strict': '#d62728',  # 严格版用红色
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


def fig1_scaling_comparison(
    output_dir: Path,
    n_vals: list[int],
    t0: float,
    alpha: float,
    k_spring: float,
    temperature: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    a: float = 1.0,
    mass: float = 1.0,
):
    """Figure 1: Case 3 vs Case 3' 标度律对比。"""
    result_approx = case3_scaling(
        n_vals=n_vals,
        t0=t0,
        alpha=alpha,
        k_spring=k_spring,
        temperature=temperature,
        well_width=well_width,
        well_depth=well_depth,
        ipr_threshold=ipr_threshold,
        r_ext=r_ext,
        a=a,
        mass=mass,
        mode="classical",
    )
    result_strict = case3_scaling_strict(
        n_vals=n_vals,
        t0=t0,
        alpha=alpha,
        k_spring=k_spring,
        temperature=temperature,
        well_width=well_width,
        well_depth=well_depth,
        ipr_threshold=ipr_threshold,
        r_ext=r_ext,
        energy_window=None,
        a=a,
        mass=mass,
        mode="classical",
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    # 拟合
    mask = result_approx.n_vals >= 40
    log_n = np.log(result_approx.n_vals[mask])
    log_y_approx = np.log(result_approx.d2_vals[mask])
    log_y_strict = np.log(result_strict.d2_vals[mask])
    slope_approx, intercept_approx = np.polyfit(log_n, log_y_approx, 1)
    slope_strict, intercept_strict = np.polyfit(log_n, log_y_strict, 1)

    ax.scatter(
        result_approx.n_vals,
        result_approx.d2_vals,
        marker="o",
        color=COLORS['approx'],
        label=f"Case 3' (Bloch approx), $\\beta$ = {-slope_approx:.2f}",
        s=60,
        zorder=3,
    )
    ax.scatter(
        result_strict.n_vals,
        result_strict.d2_vals,
        marker="s",
        color=COLORS['strict'],
        label=f"Case 3 (scattering state), $\\beta$ = {-slope_strict:.2f}",
        s=60,
        zorder=3,
    )

    # 拟合线
    n_fit = np.array(result_approx.n_vals, dtype=float)
    ax.plot(
        n_fit,
        np.exp(intercept_approx) * n_fit ** slope_approx,
        "--",
        color=COLORS['approx'],
        alpha=0.6,
    )
    ax.plot(
        n_fit,
        np.exp(intercept_strict) * n_fit ** slope_strict,
        "--",
        color=COLORS['strict'],
        alpha=0.6,
    )

    # 理论参考线
    ax.plot(
        n_fit,
        result_approx.d2_vals[0] * (n_fit[0] / n_fit),
        ":",
        color=COLORS['theory'],
        alpha=0.5,
        label=r"Theory $\propto N^{-1}$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$ (number of sites)")
    ax.set_ylabel(r"$\langle |d_{ij}|^2 \rangle$")
    ax.set_title("Case 3 scaling: Bloch approximation vs scattering state", fontsize=11)
    ax.legend(fontsize=9)

    fig.savefig(output_dir / "fig1_case3_scaling_comparison.png")
    plt.close(fig)
    print("  Figure 1: Scaling comparison - done")

    return result_approx, result_strict


def fig2_relative_difference(
    output_dir: Path,
    result_approx,
    result_strict,
) -> None:
    """Figure 2: 前因子相对差异 vs N。"""
    rel_diff = (result_strict.d2_vals - result_approx.d2_vals) / result_approx.d2_vals

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result_approx.n_vals, rel_diff * 100, "o-", color=COLORS['case3'], markersize=8)
    ax.axhline(0, linestyle="--", color=COLORS['theory'], alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("$N$ (number of sites)")
    ax.set_ylabel(r"Relative difference $(\langle d^2 \rangle_{\rm strict} - \langle d^2 \rangle_{\rm approx})/\langle d^2 \rangle_{\rm approx}$ (%)")
    ax.set_title("Prefactor difference: strict vs approximation", fontsize=11)

    fig.savefig(output_dir / "fig2_case3_relative_difference.png")
    plt.close(fig)
    print("  Figure 2: Relative difference - done")


def fig3_bloch_character(
    output_dir: Path,
    n_vals: list[int],
    t0: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    a: float = 1.0,
) -> None:
    """Figure 3: Bloch 特征收敛（使用 W_±k 指标）。"""
    result = bloch_character_vs_N(
        n_vals=n_vals,
        t0=t0,
        well_width=well_width,
        well_depth=well_depth,
        ipr_threshold=ipr_threshold,
        r_ext=r_ext,
        a=a,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.n_vals, result.overlaps, "o-", color='#9467bd', markersize=8)
    ax.axhline(1.0, linestyle="--", color=COLORS['theory'], alpha=0.5, label="Theoretical limit = 1")
    ax.set_xscale("log")
    ax.set_xlabel("$N$ (number of sites)")
    ax.set_ylabel(r"$W_{\pm k} = |\langle \psi | k \rangle|^2 + |\langle \psi | {-}k \rangle|^2$")
    ax.set_title("Bloch character convergence (standing wave metric)", fontsize=11)
    # 自动调整 ylim，但确保包含理论极限 1
    y_min = min(result.overlaps.min() * 0.95, 0.9)
    y_max = max(result.overlaps.max() * 1.05, 1.02)
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=9)

    fig.savefig(output_dir / "fig3_bloch_character.png")
    plt.close(fig)
    print("  Figure 3: Bloch character - done")


def fig4_wavefunction_realspace(
    output_dir: Path,
    n_cells: int,
    t0: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    a: float = 1.0,
) -> None:
    """Figure 4: 实空间波函数对比。"""
    # 构建含缺陷体系
    center = n_cells // 2
    onsite = build_square_well(n_cells, center=center, width=well_width, depth=well_depth)
    h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
    evals, evecs = diagonalize(h)

    # 选择散射态
    ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
    allowed = np.where(ipr_vals < ipr_threshold)[0]

    k_ext, _ = select_k_from_ratio(n_cells, r_ext, a=a)
    psi_bloch = bloch_state(n_cells, k_ext, a=a)

    selection = select_scattering_state(
        evals=evals,
        evecs=evecs,
        psi_ref=psi_bloch,
        energy_ref=dispersion(k_ext, t0, a=a),
        allowed_indices=allowed,
    )
    psi_scat = selection.state

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    sites = np.arange(n_cells)

    # 上排：波函数概率密度
    ax = axes[0]
    ax.bar(sites - 0.15, np.abs(psi_bloch) ** 2, width=0.3, color=COLORS['approx'], alpha=0.7, label=r"$|\psi_{\rm Bloch}|^2$")
    ax.bar(sites + 0.15, np.abs(psi_scat) ** 2, width=0.3, color=COLORS['strict'], alpha=0.7, label=r"$|\psi_{\rm scat}|^2$")
    ax.axvline(center, linestyle="--", color=COLORS['theory'], alpha=0.5, label="Defect center")
    ax.axvspan(center - well_width // 2, center + well_width // 2, alpha=0.1, color="yellow")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.set_title(f"Real-space wavefunction comparison ($N={n_cells}$)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(-1, n_cells)

    # 下排：差异
    ax = axes[1]
    diff = np.abs(psi_scat - psi_bloch) ** 2
    ax.bar(sites, diff, color=COLORS['case3'], alpha=0.7)
    ax.axvline(center, linestyle="--", color=COLORS['theory'], alpha=0.5)
    ax.axvspan(center - well_width // 2, center + well_width // 2, alpha=0.1, color="yellow")
    ax.set_xlabel("Site $n$")
    ax.set_ylabel(r"$|\psi_{\rm scat} - \psi_{\rm Bloch}|^2$")
    ax.set_title("Difference distribution (phase-aligned)", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_wavefunction_realspace.png")
    plt.close(fig)
    print("  Figure 4: Real-space wavefunction - done")


def fig5_wavefunction_kspace(
    output_dir: Path,
    n_cells: int,
    t0: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    a: float = 1.0,
) -> None:
    """Figure 5: 倒空间分布对比。"""
    # 构建含缺陷体系
    center = n_cells // 2
    onsite = build_square_well(n_cells, center=center, width=well_width, depth=well_depth)
    h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
    evals, evecs = diagonalize(h)

    # 选择散射态
    ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
    allowed = np.where(ipr_vals < ipr_threshold)[0]

    k_ext, _ = select_k_from_ratio(n_cells, r_ext, a=a)
    psi_bloch = bloch_state(n_cells, k_ext, a=a)

    selection = select_scattering_state(
        evals=evals,
        evecs=evecs,
        psi_ref=psi_bloch,
        energy_ref=dispersion(k_ext, t0, a=a),
        allowed_indices=allowed,
    )
    psi_scat = selection.state

    # 倒空间分布
    dist_bloch = kspace_distribution(psi_bloch, n_cells, a=a, dof_per_cell=1, shift=True)
    dist_scat = kspace_distribution(psi_scat, n_cells, a=a, dof_per_cell=1, shift=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bloch 态
    ax = axes[0]
    ax.stem(dist_bloch.k_vals / np.pi * a, dist_bloch.weights, basefmt=" ", linefmt=COLORS['approx'], markerfmt="o")
    ax.axvline(k_ext / np.pi * a, linestyle="--", color=COLORS['highlight'], alpha=0.5, label=f"$k_{{ext}}$={k_ext/np.pi:.2f}$\\pi$")
    ax.set_xlabel(r"$k$ ($\pi/a$)")
    ax.set_ylabel(r"$|c_k|^2$")
    ax.set_title("Bloch state k-space distribution", fontsize=11)
    ax.legend(fontsize=9)

    # 散射态
    ax = axes[1]
    ax.stem(dist_scat.k_vals / np.pi * a, dist_scat.weights, basefmt=" ", linefmt=COLORS['strict'], markerfmt="s")
    ax.axvline(k_ext / np.pi * a, linestyle="--", color=COLORS['approx'], alpha=0.5, label=f"$k_{{ext}}$={k_ext/np.pi:.2f}$\\pi$")
    ax.set_xlabel(r"$k$ ($\pi/a$)")
    ax.set_ylabel(r"$|c_k|^2$")
    ax.set_title("Scattering state k-space distribution", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_wavefunction_kspace.png")
    plt.close(fig)
    print("  Figure 5: k-space wavefunction - done")


def main() -> None:
    """执行全部 Case 3 对比可视化。"""
    setup_publication_style()
    output_dir = ensure_output_dir()

    # 参数（从 40 开始以确保有足够的延展态）
    n_vals = [40, 60, 80, 120, 160, 240, 320, 480, 640]
    t0 = 1.0
    alpha = 0.5
    k_spring = 1.0
    temperature = 300.0
    well_width = 5
    well_depth = -1.5
    ipr_threshold = 0.05
    r_ext = 1 / 10
    a = 1.0
    mass = 1.0

    print(f"Output directory: {output_dir}\n")

    print("Generating Figure 1: Scaling comparison...")
    result_approx, result_strict = fig1_scaling_comparison(
        output_dir=output_dir,
        n_vals=n_vals,
        t0=t0,
        alpha=alpha,
        k_spring=k_spring,
        temperature=temperature,
        well_width=well_width,
        well_depth=well_depth,
        ipr_threshold=ipr_threshold,
        r_ext=r_ext,
        a=a,
        mass=mass,
    )

    print("Generating Figure 2: Relative difference...")
    fig2_relative_difference(output_dir, result_approx, result_strict)

    print("Generating Figure 3: Bloch character...")
    fig3_bloch_character(
        output_dir=output_dir,
        n_vals=n_vals,
        t0=t0,
        well_width=well_width,
        well_depth=well_depth,
        ipr_threshold=ipr_threshold,
        r_ext=r_ext,
        a=a,
    )

    print("Generating Figure 4: Real-space wavefunction...")
    fig4_wavefunction_realspace(
        output_dir=output_dir,
        n_cells=80,
        t0=t0,
        well_width=well_width,
        well_depth=well_depth,
        ipr_threshold=ipr_threshold,
        r_ext=r_ext,
        a=a,
    )

    print("Generating Figure 5: k-space wavefunction...")
    fig5_wavefunction_kspace(
        output_dir=output_dir,
        n_cells=80,
        t0=t0,
        well_width=well_width,
        well_depth=well_depth,
        ipr_threshold=ipr_threshold,
        r_ext=r_ext,
        a=a,
    )

    print("\nCase 3 comparison visualization complete!")


if __name__ == "__main__":
    main()
