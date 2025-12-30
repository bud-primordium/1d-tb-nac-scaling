"""耦合密度可视化：三种 Case 的实空间分布与幅度标度。

Figure 6: 三种 Case 的耦合密度实空间分布
Figure 7: 耦合密度幅度标度 (max_n |D(n)| vs N)

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

from src.coupling_density import coupling_density_realspace
from src.diagnostics import ipr, select_bloch_states, select_k_from_ratio
from src.electron_phonon import dh_dq_monatomic
from src.tb_electron_1band import (
    bloch_state,
    build_hamiltonian,
    build_square_well,
    diagonalize,
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


# =============================================================================
# 固定物理 q 的选择
# =============================================================================

# 使用固定的物理 q₀ (不随 N 变化)
Q0_FIXED = 0.2 * np.pi  # 固定 q₀ = 0.2π/a


def nearest_q_on_grid(n_cells: int, q_target: float, a: float = 1.0) -> float:
    """返回离散 q 网格上最接近 q_target 的值。"""
    q_grid = 2.0 * np.pi * np.arange(n_cells) / (n_cells * a)
    idx = np.argmin(np.abs(q_grid - q_target))
    return q_grid[idx]


# =============================================================================
# 耦合密度计算函数
# =============================================================================

def compute_case1_density(
    n_cells: int,
    r1: float,
    r2: float,
    alpha: float,
    a: float = 1.0,
    mass: float = 1.0,
    use_fixed_q: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Case 1: 延展-延展的耦合密度。"""
    k1, k2, m1, m2 = select_bloch_states(n_cells, r1, r2, a=a)
    psi_i = bloch_state(n_cells, k1, a=a)
    psi_j = bloch_state(n_cells, k2, a=a)

    if use_fixed_q:
        # 使用固定 q
        q0 = nearest_q_on_grid(n_cells, Q0_FIXED, a=a)
    else:
        # 满足动量守恒的 q
        q_index = (m1 - m2) % n_cells
        q0 = 2.0 * np.pi * q_index / (n_cells * a)

    dh = dh_dq_monatomic(n_cells, q0, alpha, a=a, mass=mass)
    density = coupling_density_realspace(psi_i, psi_j, dh)

    return np.arange(n_cells), density


def compute_case2_density(
    n_cells: int,
    t0: float,
    alpha: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    a: float = 1.0,
    mass: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Case 2: 局域-局域的耦合密度。使用固定 q₀。"""
    center = n_cells // 2
    onsite = build_square_well(n_cells, center=center, width=well_width, depth=well_depth)
    h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
    evals, evecs = diagonalize(h)

    ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
    localized_indices = np.where(ipr_vals >= ipr_threshold)[0]
    if localized_indices.size < 2:
        order = np.argsort(ipr_vals)[::-1]
        idx1, idx2 = order[0], order[1]
    else:
        local_ipr = ipr_vals[localized_indices]
        order = np.argsort(local_ipr)[::-1]
        idx1, idx2 = localized_indices[order[0]], localized_indices[order[1]]

    psi_i = evecs[:, idx1]
    psi_j = evecs[:, idx2]

    # 使用固定 q₀ (取最近的离散值)
    q0 = nearest_q_on_grid(n_cells, Q0_FIXED, a=a)
    dh = dh_dq_monatomic(n_cells, q=q0, alpha=alpha, a=a, mass=mass)
    density = coupling_density_realspace(psi_i, psi_j, dh)

    return np.arange(n_cells), density


def compute_case3_density(
    n_cells: int,
    t0: float,
    alpha: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    a: float = 1.0,
    mass: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Case 3: 局域-延展的耦合密度。使用固定 q₀。"""
    center = n_cells // 2
    onsite = build_square_well(n_cells, center=center, width=well_width, depth=well_depth)
    h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
    evals, evecs = diagonalize(h)

    ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
    localized_indices = np.where(ipr_vals >= ipr_threshold)[0]
    if localized_indices.size == 0:
        raise ValueError("未找到局域态")
    loc_idx = localized_indices[np.argmax(ipr_vals[localized_indices])]
    psi_loc = evecs[:, loc_idx]

    k_ext, _ = select_k_from_ratio(n_cells, r_ext, a=a)
    psi_ext = bloch_state(n_cells, k_ext, a=a)

    # 使用固定 q₀ (取最近的离散值)
    q0 = nearest_q_on_grid(n_cells, Q0_FIXED, a=a)
    dh = dh_dq_monatomic(n_cells, q=q0, alpha=alpha, a=a, mass=mass)
    density = coupling_density_realspace(psi_loc, psi_ext, dh)

    return np.arange(n_cells), density


# =============================================================================
# 图表生成函数
# =============================================================================

def fig6_realspace_distribution(
    output_dir: Path,
    n_cells: int,
    t0: float,
    alpha: float,
    r1: float,
    r2: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    a: float = 1.0,
    mass: float = 1.0,
) -> None:
    """Figure 6: 三种 Case 的耦合密度实空间分布。"""
    center = n_cells // 2

    # 计算三种 Case 的耦合密度
    sites1, density1 = compute_case1_density(n_cells, r1, r2, alpha, a=a, mass=mass)
    sites2, density2 = compute_case2_density(
        n_cells, t0, alpha, well_width, well_depth, ipr_threshold, a=a, mass=mass
    )
    sites3, density3 = compute_case3_density(
        n_cells, t0, alpha, well_width, well_depth, ipr_threshold, r_ext, a=a, mass=mass
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Case 1
    ax = axes[0]
    ax.bar(sites1, np.abs(density1), color=COLORS['case1'], alpha=0.8, width=0.8)
    ax.set_ylabel(r'$|D(n)|$')
    ax.set_title(r'(a) Case 1 (Ext-Ext): $|D(n)| \sim 1/N$, uniform distribution', fontsize=11)
    max_d1 = np.max(np.abs(density1))
    ax.text(0.97, 0.85, f"max|D| = {max_d1:.2e}", transform=ax.transAxes,
            fontsize=9, ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Case 2
    ax = axes[1]
    ax.bar(sites2, np.abs(density2), color=COLORS['case2'], alpha=0.8, width=0.8)
    ax.axvline(center, linestyle="--", color=COLORS['theory'], alpha=0.6)
    ax.axvspan(center - well_width // 2, center + well_width // 2, alpha=0.1, color="yellow")
    ax.set_ylabel(r'$|D(n)|$')
    ax.set_title(r'(b) Case 2 (Loc-Loc): $|D(n)| \sim O(1)$, localized near defect', fontsize=11)
    max_d2 = np.max(np.abs(density2))
    ax.text(0.97, 0.85, f"max|D| = {max_d2:.2e}", transform=ax.transAxes,
            fontsize=9, ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Case 3
    ax = axes[2]
    ax.bar(sites3, np.abs(density3), color=COLORS['case3'], alpha=0.8, width=0.8)
    ax.axvline(center, linestyle="--", color=COLORS['theory'], alpha=0.6)
    ax.axvspan(center - well_width // 2, center + well_width // 2, alpha=0.1, color="yellow")
    ax.set_xlabel("Site $n$")
    ax.set_ylabel(r'$|D(n)|$')
    ax.set_title(r'(c) Case 3 (Loc-Ext): $|D(n)| \sim 1/\sqrt{N}$, localized near defect', fontsize=11)
    max_d3 = np.max(np.abs(density3))
    ax.text(0.97, 0.85, f"max|D| = {max_d3:.2e}", transform=ax.transAxes,
            fontsize=9, ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "fig6_coupling_density_realspace.png")
    plt.close(fig)
    print("  Figure 6: Coupling density (real space) - done")


def fig7_amplitude_scaling(
    output_dir: Path,
    n_vals: list[int],
    t0: float,
    alpha: float,
    r1: float,
    r2: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    a: float = 1.0,
    mass: float = 1.0,
) -> None:
    """Figure 7: 耦合密度幅度标度 (max_n |D(n)| vs N)。

    使用固定物理 q₀，确保标度律正确反映波函数性质。
    """
    max_d1_vals = []
    max_d2_vals = []
    max_d3_vals = []

    for n_cells in n_vals:
        # Case 1: 使用固定 q（与 Case 2/3 一致，便于对比）
        _, density1 = compute_case1_density(
            n_cells, r1, r2, alpha, a=a, mass=mass, use_fixed_q=True
        )
        max_d1_vals.append(np.max(np.abs(density1)))

        # Case 2
        _, density2 = compute_case2_density(
            n_cells, t0, alpha, well_width, well_depth, ipr_threshold, a=a, mass=mass
        )
        max_d2_vals.append(np.max(np.abs(density2)))

        # Case 3
        _, density3 = compute_case3_density(
            n_cells, t0, alpha, well_width, well_depth, ipr_threshold, r_ext, a=a, mass=mass
        )
        max_d3_vals.append(np.max(np.abs(density3)))

    n_arr = np.array(n_vals, dtype=float)
    max_d1 = np.array(max_d1_vals)
    max_d2 = np.array(max_d2_vals)
    max_d3 = np.array(max_d3_vals)

    # 拟合
    mask = n_arr >= 40
    log_n = np.log(n_arr[mask])

    slope1, _ = np.polyfit(log_n, np.log(max_d1[mask]), 1)
    slope2, _ = np.polyfit(log_n, np.log(max_d2[mask]), 1)
    slope3, _ = np.polyfit(log_n, np.log(max_d3[mask]), 1)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(n_arr, max_d1, s=60, marker="o", color=COLORS['case1'],
               label=f"Case 1 (Ext-Ext): slope = {slope1:.2f} (theory: -1)", zorder=3)
    ax.scatter(n_arr, max_d2, s=60, marker="s", color=COLORS['case2'],
               label=f"Case 2 (Loc-Loc): slope = {slope2:.2f} (theory: 0)", zorder=3)
    ax.scatter(n_arr, max_d3, s=60, marker="^", color=COLORS['case3'],
               label=f"Case 3 (Loc-Ext): slope = {slope3:.2f} (theory: -0.5)", zorder=3)

    # 理论线
    n_fit = np.linspace(n_arr.min() * 0.9, n_arr.max() * 1.1, 100)
    # Case 1: N^{-1}
    A1 = max_d1[0] * n_arr[0]
    ax.plot(n_fit, A1 / n_fit, "--", color=COLORS['case1'], alpha=0.5, lw=1.5)
    # Case 2: N^0 (常数)
    A2 = np.mean(max_d2)
    ax.plot(n_fit, np.full_like(n_fit, A2), "--", color=COLORS['case2'], alpha=0.5, lw=1.5)
    # Case 3: N^{-0.5}
    A3 = max_d3[0] * np.sqrt(n_arr[0])
    ax.plot(n_fit, A3 / np.sqrt(n_fit), "--", color=COLORS['case3'], alpha=0.5, lw=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$ (number of sites)")
    ax.set_ylabel(r"$\max_n |D(n)|$")
    ax.set_title(f"Coupling density amplitude scaling (fixed $q_0 = {Q0_FIXED/np.pi:.1f}\\pi/a$)", fontsize=11)
    ax.legend(fontsize=9, loc='lower left')

    plt.tight_layout()
    fig.savefig(output_dir / "fig7_coupling_density_scaling.png")
    plt.close(fig)
    print("  Figure 7: Coupling density scaling - done")


def main() -> None:
    """执行全部耦合密度可视化。"""
    setup_publication_style()
    output_dir = ensure_output_dir()

    # 参数
    n_vals = [20, 40, 60, 80, 120, 160, 240, 320]
    t0 = 1.0
    alpha = 0.5
    r1 = 1 / 10
    r2 = 3 / 10
    well_width = 5
    well_depth = -1.5
    ipr_threshold = 0.05
    r_ext = 1 / 10
    a = 1.0
    mass = 1.0

    print(f"Output directory: {output_dir}\n")

    print("Generating Figure 6: Coupling density (real space)...")
    fig6_realspace_distribution(
        output_dir=output_dir,
        n_cells=80,
        t0=t0,
        alpha=alpha,
        r1=r1,
        r2=r2,
        well_width=well_width,
        well_depth=well_depth,
        ipr_threshold=ipr_threshold,
        r_ext=r_ext,
        a=a,
        mass=mass,
    )

    print("Generating Figure 7: Coupling density scaling...")
    fig7_amplitude_scaling(
        output_dir=output_dir,
        n_vals=n_vals,
        t0=t0,
        alpha=alpha,
        r1=r1,
        r2=r2,
        well_width=well_width,
        well_depth=well_depth,
        ipr_threshold=ipr_threshold,
        r_ext=r_ext,
        a=a,
        mass=mass,
    )

    print(f"\nCoupling density visualization complete!")


if __name__ == "__main__":
    main()
