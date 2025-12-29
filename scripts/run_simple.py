"""运行简化版实验并输出图表。"""

from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import matplotlib.pyplot as plt

from src.diagnostics import (
    find_localized_states,
    ipr,
    select_bloch_states,
    select_k_from_ratio,
    top_ipr_indices,
)
from src.electron_phonon import g_monatomic
from src.experiments_simple import case1_scaling, case2_scaling, case3_scaling, folded_phonon_demo
from src.lattice import q_grid
from src.phonon_1atom import displacement_from_q
from src.tb_electron_1band import build_hamiltonian, build_square_well, diagonalize
from src.visualization import plot_scaling, save_figure, setup_plot_style


def ensure_dirs() -> None:
    """确保输出目录存在。"""
    pathlib.Path("results/figures").mkdir(parents=True, exist_ok=True)


def plot_normalization_simple(n_vals: list[int], a: float, mass: float) -> None:
    """归一化自检：sum |u_n|^2 是否随 N 保持常数。"""
    norms = []
    for n_cells in n_vals:
        q_vals = np.array([2.0 * np.pi / (n_cells * a)])
        q_amp = np.array([1.0 + 0.0j])
        u = displacement_from_q(q_vals, q_amp, n_cells, a=a, mass=mass)
        norms.append(np.sum(np.abs(u) ** 2))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n_vals, norms, marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\sum_n |u_n|^2$")
    ax.set_title("简化版归一化自检")
    save_figure(fig, "results/figures/S6_normalization_simple")


def plot_case2_wavefunctions(
    n_cells: int,
    t0: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
) -> None:
    """Case 2 波函数与 IPR 可视化。"""
    onsite = build_square_well(n_cells, center=n_cells // 2, width=well_width, depth=well_depth)
    h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
    evals, evecs = diagonalize(h)
    localized = find_localized_states(evecs, ipr_threshold)
    if localized.size >= 2:
        ipr_vals = np.array([ipr(evecs[:, i]) for i in localized])
        order = np.argsort(ipr_vals)[::-1]
        idx1, idx2 = localized[order[:2]]
    else:
        idx1, idx2 = top_ipr_indices(evecs, count=2)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.abs(evecs[:, idx1]) ** 2, label=f"L1 (E={evals[idx1]:.3f})")
    ax.plot(np.abs(evecs[:, idx2]) ** 2, label=f"L2 (E={evals[idx2]:.3f})")
    ax.set_xlabel("格点")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.set_title("Case 2 局域态波函数")
    ax.legend()
    save_figure(fig, "results/figures/S3_case2_wavefunction")

    fig, ax = plt.subplots(figsize=(5, 3))
    ipr_all = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
    ax.scatter(evals, ipr_all, s=15, alpha=0.8)
    ax.set_xlabel("能量")
    ax.set_ylabel("IPR")
    ax.set_title("Case 2 IPR 分布")
    save_figure(fig, "results/figures/S3_case2_ipr")


def plot_case3_gq_distribution(
    n_cells: int,
    t0: float,
    alpha: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    a: float,
    mass: float,
) -> None:
    """Case 3 的 g(q) 分布示意。"""
    onsite = build_square_well(n_cells, center=n_cells // 2, width=well_width, depth=well_depth)
    h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
    evals, evecs = diagonalize(h)
    ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
    localized = np.where(ipr_vals >= ipr_threshold)[0]
    if localized.size == 0:
        raise ValueError("未找到局域态，无法绘制 Case 3 g(q)")
    loc_idx = localized[np.argmax(ipr_vals[localized])]
    psi_i = evecs[:, loc_idx]
    k_ext, _ = select_k_from_ratio(n_cells, r_ext, a=a)
    psi_j = np.exp(1j * k_ext * np.arange(n_cells) * a) / np.sqrt(n_cells)
    q_vals = q_grid(n_cells, a=a)
    g_vals = np.array(
        [g_monatomic(psi_i, psi_j, n_cells, q, alpha, a=a, mass=mass) for q in q_vals]
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(q_vals, np.abs(g_vals))
    ax.set_xlabel("q")
    ax.set_ylabel(r"$|g(q)|$")
    ax.set_title("Case 3 的 g(q) 分布")
    save_figure(fig, "results/figures/S4_case3_gq")


def plot_debug_case1(
    n_vals: list[int],
    r1: float,
    r2: float,
    t0: float,
    alpha: float,
    a: float,
    mass: float,
) -> None:
    """Case 1′ 的调试图：ΔE、单模 |g|、有效模式数。"""
    delta_e_list = []
    gq0_list = []
    mode_count = []
    for n_cells in n_vals:
        k1, k2, m1, m2 = select_bloch_states(n_cells, r1, r2, a=a)
        psi_i = np.exp(1j * k1 * np.arange(n_cells) * a) / np.sqrt(n_cells)
        psi_j = np.exp(1j * k2 * np.arange(n_cells) * a) / np.sqrt(n_cells)
        q_index = (m1 - m2) % n_cells
        q0 = 2.0 * np.pi * q_index / (n_cells * a)
        delta_e = 2.0 * t0 * (np.cos(k2 * a) - np.cos(k1 * a))
        delta_e_list.append(delta_e)

        g_vals = []
        for q in q_grid(n_cells, a=a):
            g_vals.append(g_monatomic(psi_i, psi_j, n_cells, q, alpha, a=a, mass=mass))
        g_vals = np.array(g_vals)
        gq0_list.append(np.abs(g_monatomic(psi_i, psi_j, n_cells, q0, alpha, a=a, mass=mass)) ** 2)
        threshold = 1e-6 * np.max(np.abs(g_vals))
        mode_count.append(np.sum(np.abs(g_vals) > threshold))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n_vals, delta_e_list, marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\Delta E$")
    ax.set_title("Case 1′ ΔE(N)")
    save_figure(fig, "results/figures/S2_debug_delta_e")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n_vals, gq0_list, marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$|g(q_0)|^2$")
    ax.set_title("Case 1′ 单模耦合强度")
    save_figure(fig, "results/figures/S2_debug_gq0")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n_vals, mode_count, marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel("有效模式数")
    ax.set_title("Case 1′ 参与模式数")
    save_figure(fig, "results/figures/S2_debug_modecount")


def main() -> None:
    """执行简化版全部实验。"""
    ensure_dirs()
    setup_plot_style()

    n_vals = [20, 40, 60, 80, 120, 160, 240, 320, 480, 640]
    r1 = 1 / 10
    r2 = 3 / 10
    t0 = 1.0
    alpha = 0.5
    k_spring = 1.0
    temperature = 300.0
    a = 1.0
    mass = 1.0

    plot_normalization_simple(n_vals, a=a, mass=mass)

    result_case1 = case1_scaling(
        n_vals=n_vals,
        r1=r1,
        r2=r2,
        t0=t0,
        alpha=alpha,
        k_spring=k_spring,
        temperature=temperature,
        a=a,
        mass=mass,
        mode="classical",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_scaling(ax, result_case1.n_vals, result_case1.d2_vals, label="Case 1′", color="tab:blue")
    save_figure(fig, "results/figures/S2_case1_scaling")

    plot_debug_case1(n_vals, r1, r2, t0, alpha, a, mass)

    q_branches, g_vals = folded_phonon_demo(
        n_cells=40,
        r1=r1,
        r2=r2,
        m_supercell=5,
        t0=t0,
        alpha=alpha,
        a=a,
        mass=mass,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(q_branches)), np.abs(g_vals))
    ax.set_xlabel("折叠分支索引")
    ax.set_ylabel(r"$|g(q)|$")
    ax.set_title("折叠分支矩阵元对比")
    save_figure(fig, "results/figures/S1_folded_phonon")

    result_case2 = case2_scaling(
        n_vals=n_vals,
        t0=t0,
        alpha=alpha,
        k_spring=k_spring,
        temperature=temperature,
        well_width=5,
        well_depth=-1.5,
        ipr_threshold=0.05,
        a=a,
        mass=mass,
        mode="classical",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_scaling(ax, result_case2.n_vals, result_case2.d2_vals, label="Case 2", color="tab:orange")
    save_figure(fig, "results/figures/S3_case2_scaling")
    plot_case2_wavefunctions(
        n_cells=80,
        t0=t0,
        well_width=5,
        well_depth=-1.5,
        ipr_threshold=0.05,
    )

    result_case3 = case3_scaling(
        n_vals=n_vals,
        t0=t0,
        alpha=alpha,
        k_spring=k_spring,
        temperature=temperature,
        well_width=5,
        well_depth=-1.5,
        ipr_threshold=0.05,
        r_ext=r1,
        a=a,
        mass=mass,
        mode="classical",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_scaling(ax, result_case3.n_vals, result_case3.d2_vals, label="Case 3", color="tab:green")
    save_figure(fig, "results/figures/S4_case3_scaling")
    plot_case3_gq_distribution(
        n_cells=80,
        t0=t0,
        alpha=alpha,
        well_width=5,
        well_depth=-1.5,
        ipr_threshold=0.05,
        r_ext=r1,
        a=a,
        mass=mass,
    )


if __name__ == "__main__":
    main()
