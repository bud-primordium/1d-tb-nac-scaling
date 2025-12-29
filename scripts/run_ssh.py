"""运行 SSH 完整版实验并输出图表。"""

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

from src.diagnostics import ipr
from src.electron_phonon import g_ssh_diatomic
from src.experiments_ssh import case1_scaling, case2_scaling, case3_scaling, folded_phonon_demo, select_gap_k
from src.lattice import q_grid
from src.phonon_diatomic import diatomic_modes, diatomic_modes_grid, displacement_from_q
from src.ssh_electron import build_hamiltonian, build_square_well, diagonalize, bloch_eigensystem, bloch_state
from src.visualization import plot_scaling, save_figure, setup_plot_style


def ensure_dirs() -> None:
    """确保输出目录存在。"""
    pathlib.Path("results/figures").mkdir(parents=True, exist_ok=True)


def plot_normalization_ssh(n_vals: list[int], a: float, mass_a: float, mass_b: float) -> None:
    """归一化自检：双原子链的位移归一化。"""
    norms = []
    for n_cells in n_vals:
        q_vals = np.array([0.0])
        omegas, evecs = diatomic_modes_grid(q_vals, k_spring=1.0, mass_a=mass_a, mass_b=mass_b, a=a)
        q_amp = np.array([[0.0, 1.0]])
        disp = displacement_from_q(q_vals, q_amp, evecs, n_cells, a=a, mass_a=mass_a, mass_b=mass_b)
        norm = np.sum(np.abs(disp[0]) ** 2) + np.sum(np.abs(disp[1]) ** 2)
        norms.append(norm)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n_vals, norms, marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\sum |u|^2$")
    ax.set_title("SSH 归一化自检")
    save_figure(fig, "results/figures/F5_normalization_ssh")


def plot_debug_case1(
    n_vals: list[int],
    t0: float,
    delta_t: float,
    alpha: float,
    k_spring: float,
    a: float,
    mass_a: float,
    mass_b: float,
) -> None:
    """Case 1 的调试图：ΔE、单模 |g|、有效模式数。"""
    delta_e_list = []
    gq0_list = []
    mode_count = []
    for n_cells in n_vals:
        k_gap = select_gap_k(n_cells, t0=t0, delta_t=delta_t, a=a)
        evals_k, evecs_k = bloch_eigensystem(k_gap, t0=t0, delta_t=delta_t, a=a)
        psi_v = bloch_state(n_cells, k_gap, evecs_k[:, 0], a=a)
        psi_c = bloch_state(n_cells, k_gap, evecs_k[:, 1], a=a)
        delta_e = float(evals_k[1] - evals_k[0])
        delta_e_list.append(delta_e)

        omega_q, evec_q = diatomic_modes(0.0, k_spring, mass_a, mass_b, a=a)
        mode_idx = int(np.argmax(omega_q))
        g_val = g_ssh_diatomic(
            psi_v,
            psi_c,
            n_cells,
            q=0.0,
            evec=evec_q[:, mode_idx],
            alpha=alpha,
            a=a,
            mass_a=mass_a,
            mass_b=mass_b,
        )
        gq0_list.append(np.abs(g_val) ** 2)

        q_vals = q_grid(n_cells, a=a)
        omegas, evecs = diatomic_modes_grid(q_vals, k_spring, mass_a, mass_b, a=a)
        g_vals = np.zeros((q_vals.shape[0], 2), dtype=complex)
        for idx, q in enumerate(q_vals):
            for mode in range(2):
                g_vals[idx, mode] = g_ssh_diatomic(
                    psi_v,
                    psi_c,
                    n_cells,
                    q=q,
                    evec=evecs[idx, :, mode],
                    alpha=alpha,
                    a=a,
                    mass_a=mass_a,
                    mass_b=mass_b,
                )
        g_flat = np.abs(g_vals).ravel()
        threshold = 1e-6 * np.max(g_flat)
        mode_count.append(int(np.sum(g_flat > threshold)))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n_vals, delta_e_list, marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\Delta E$")
    ax.set_title("SSH Case 1 ΔE(N)")
    save_figure(fig, "results/figures/F2_debug_delta_e")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n_vals, gq0_list, marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$|g(q=0)|^2$")
    ax.set_title("SSH 单模耦合强度")
    save_figure(fig, "results/figures/F2_debug_gq0")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n_vals, mode_count, marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel("有效模式数")
    ax.set_title("SSH Case 1 参与模式数")
    save_figure(fig, "results/figures/F2_debug_modecount")


def main() -> None:
    """执行 SSH 完整版全部实验。"""
    ensure_dirs()
    setup_plot_style()

    n_vals = [40, 80, 160, 320]
    t0 = 1.0
    delta_t = 0.2
    alpha = 0.5
    k_spring = 1.0
    temperature = 300.0
    r_ext = 1 / 10
    a = 1.0
    mass_a = 1.0
    mass_b = 1.0

    plot_normalization_ssh(n_vals, a=a, mass_a=mass_a, mass_b=mass_b)

    result_case1 = case1_scaling(
        n_vals=n_vals,
        t0=t0,
        delta_t=delta_t,
        alpha=alpha,
        k_spring=k_spring,
        temperature=temperature,
        a=a,
        mass_a=mass_a,
        mass_b=mass_b,
        mode="classical",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_scaling(ax, result_case1.n_vals, result_case1.d2_vals, label="SSH Case 1", color="tab:blue")
    save_figure(fig, "results/figures/F2_case1_ssh")

    plot_debug_case1(n_vals, t0, delta_t, alpha, k_spring, a, mass_a, mass_b)

    result_case2 = case2_scaling(
        n_vals=n_vals,
        t0=t0,
        delta_t=delta_t,
        alpha=alpha,
        k_spring=k_spring,
        temperature=temperature,
        well_width=5,
        well_depth=-1.5,
        ipr_threshold=0.05,
        a=a,
        mass_a=mass_a,
        mass_b=mass_b,
        mode="classical",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_scaling(ax, result_case2.n_vals, result_case2.d2_vals, label="SSH Case 2", color="tab:orange")
    save_figure(fig, "results/figures/F3_case2_ssh")

    result_case3 = case3_scaling(
        n_vals=n_vals,
        t0=t0,
        delta_t=delta_t,
        alpha=alpha,
        k_spring=k_spring,
        temperature=temperature,
        well_width=5,
        well_depth=-1.5,
        ipr_threshold=0.05,
        r_ext=r_ext,
        ext_band=0,
        a=a,
        mass_a=mass_a,
        mass_b=mass_b,
        mode="classical",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_scaling(ax, result_case3.n_vals, result_case3.d2_vals, label="SSH Case 3", color="tab:green")
    save_figure(fig, "results/figures/F4_case3_ssh")

    g_vals, _, _ = folded_phonon_demo(
        n_cells=2,
        t0=t0,
        delta_t=delta_t,
        alpha=alpha,
        k_spring=k_spring,
        a=a,
        mass_a=mass_a,
        mass_b=mass_b,
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([0, 1], np.abs(g_vals))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["模式 I", "模式 II"])
    ax.set_ylabel(r"$|g|$")
    ax.set_title("折叠纹理耦合对比")
    save_figure(fig, "results/figures/F1_folded_ssh")


if __name__ == "__main__":
    main()
