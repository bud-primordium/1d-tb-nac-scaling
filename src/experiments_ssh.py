"""SSH 完整版实验：两带 + 双原子链。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .diagnostics import ipr, select_k_from_ratio, top_ipr_indices
from .electron_phonon import g_ssh_diatomic, g_ssh_diatomic_grid, g_ssh_from_displacements
from .lattice import k_grid, q_grid
from .nac import mean_square_nac, qdot_variance
from .phonon_diatomic import diatomic_modes, diatomic_modes_grid
from .ssh_electron import (
    bloch_eigensystem,
    bloch_state,
    build_hamiltonian,
    build_square_well,
    diagonalize,
)


@dataclass
class CaseResult:
    """单个 Case 的结果容器。"""

    n_vals: np.ndarray
    d2_vals: np.ndarray
    delta_e_vals: np.ndarray


def select_gap_k(
    n_cells: int,
    t0: float,
    delta_t: float,
    a: float = 1.0,
) -> float:
    """在离散 k 网格上寻找最小带隙对应的 k。"""
    k_vals = k_grid(n_cells, a=a)
    gaps = []
    for k in k_vals:
        evals_k, _ = bloch_eigensystem(k, t0=t0, delta_t=delta_t, a=a)
        gaps.append(evals_k[1] - evals_k[0])
    gaps = np.array(gaps, dtype=float)
    min_idx = int(np.argmin(gaps))
    return float(k_vals[min_idx])


def case1_scaling(
    n_vals: List[int],
    t0: float,
    delta_t: float,
    alpha: float,
    k_spring: float,
    temperature: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
    mode: str = "classical",
    r_k: float = 0.1,
) -> CaseResult:
    """Case 1：同 k 的 VBM–CBM（延展态对）。

    注意：避免使用 BZ 边界 k=π/a，因为该处 VBM-CBM 跃迁被对称性禁止。
    使用 r_k 参数指定 k = 2π * r_k / a 的位置。
    """
    d2_vals = []
    delta_e_vals = []
    for n_cells in n_vals:
        # 使用远离 BZ 边界的 k 点，避免对称性禁止
        k_ext, _ = select_k_from_ratio(n_cells, r_k, a=a)
        evals_k, evecs_k = bloch_eigensystem(k_ext, t0=t0, delta_t=delta_t, a=a)
        psi_v = bloch_state(n_cells, k_ext, evecs_k[:, 0], a=a)
        psi_c = bloch_state(n_cells, k_ext, evecs_k[:, 1], a=a)
        delta_e = float(evals_k[1] - evals_k[0])

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
        qdot_var = qdot_variance(np.array([omega_q[mode_idx]]), temperature, mode=mode)
        d2 = mean_square_nac(np.array([g_val]), delta_e, qdot_var)

        d2_vals.append(d2)
        delta_e_vals.append(delta_e)

    return CaseResult(
        n_vals=np.array(n_vals, dtype=int),
        d2_vals=np.array(d2_vals, dtype=float),
        delta_e_vals=np.array(delta_e_vals, dtype=float),
    )


def folded_phonon_demo(
    n_cells: int,
    t0: float,
    delta_t: float,
    alpha: float,
    k_spring: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """折叠声子验证：比较两种位移纹理的一阶耦合。

    演示只有光学型模式（胞内相对运动）有非零耦合，
    而声学型模式（胞内同相运动）耦合为零。
    """
    _ = k_spring
    # 对于 n_cells=2，只有 k=0 和 k=π/a 两个点
    # k=π/a 被对称性禁止，所以用 k=π/(2a)，这需要 n_cells≥4
    # 对于 n_cells=2，直接用 k=π/(4a) 作为连续近似
    if n_cells == 2:
        k_ext = np.pi / (4.0 * a)  # 使用非离散点，仅用于演示
    else:
        k_ext, _ = select_k_from_ratio(n_cells, 0.25, a=a)  # k = π/(2a)

    evals_k, evecs_k = bloch_eigensystem(k_ext, t0=t0, delta_t=delta_t, a=a)
    psi_v = bloch_state(n_cells, k_ext, evecs_k[:, 0], a=a)
    psi_c = bloch_state(n_cells, k_ext, evecs_k[:, 1], a=a)

    if n_cells != 2:
        raise ValueError("折叠纹理示例建议使用 n_cells=2 的超胞")

    u_a_mode1 = np.array([1.0, 1.0], dtype=float)
    u_b_mode1 = np.array([-1.0, -1.0], dtype=float)
    u_a_mode2 = np.array([1.0, -1.0], dtype=float)
    u_b_mode2 = np.array([-1.0, 1.0], dtype=float)

    g_mode1 = g_ssh_from_displacements(psi_v, psi_c, u_a_mode1, u_b_mode1, alpha=alpha)
    g_mode2 = g_ssh_from_displacements(psi_v, psi_c, u_a_mode2, u_b_mode2, alpha=alpha)

    return np.array([g_mode1, g_mode2]), np.array([u_a_mode1, u_a_mode2]), np.array([u_b_mode1, u_b_mode2])


def case2_scaling(
    n_vals: List[int],
    t0: float,
    delta_t: float,
    alpha: float,
    k_spring: float,
    temperature: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
    mode: str = "classical",
) -> CaseResult:
    """Case 2：局域-局域。"""
    d2_vals = []
    delta_e_vals = []
    for n_cells in n_vals:
        onsite = build_square_well(n_cells, center=n_cells // 2, width=well_width, depth=well_depth)
        h = build_hamiltonian(n_cells, t0=t0, delta_t=delta_t, onsite=onsite, pbc=True)
        evals, evecs = diagonalize(h)

        ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
        localized = np.where(ipr_vals >= ipr_threshold)[0]
        if localized.size >= 2:
            local_ipr = ipr_vals[localized]
            order = np.argsort(local_ipr)[::-1]
            idx1, idx2 = localized[order[:2]]
        else:
            idx1, idx2 = top_ipr_indices(evecs, count=2)
        psi_i = evecs[:, idx1]
        psi_j = evecs[:, idx2]
        delta_e = float(evals[idx2] - evals[idx1])

        q_vals = q_grid(n_cells, a=a)
        omegas, evecs_q = diatomic_modes_grid(q_vals, k_spring, mass_a, mass_b, a=a)
        g_vals = g_ssh_diatomic_grid(
            psi_i,
            psi_j,
            n_cells,
            q_vals,
            evecs_q,
            alpha,
            a=a,
            mass_a=mass_a,
            mass_b=mass_b,
        )
        qdot_var = qdot_variance(omegas, temperature, mode=mode)
        d2 = mean_square_nac(g_vals, delta_e, qdot_var)

        d2_vals.append(d2)
        delta_e_vals.append(delta_e)

    return CaseResult(
        n_vals=np.array(n_vals, dtype=int),
        d2_vals=np.array(d2_vals, dtype=float),
        delta_e_vals=np.array(delta_e_vals, dtype=float),
    )


def case3_scaling(
    n_vals: List[int],
    t0: float,
    delta_t: float,
    alpha: float,
    k_spring: float,
    temperature: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    ext_band: int = 0,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
    mode: str = "classical",
) -> CaseResult:
    """Case 3：局域-延展。"""
    d2_vals = []
    delta_e_vals = []
    for n_cells in n_vals:
        onsite = build_square_well(n_cells, center=n_cells // 2, width=well_width, depth=well_depth)
        h = build_hamiltonian(n_cells, t0=t0, delta_t=delta_t, onsite=onsite, pbc=True)
        evals, evecs = diagonalize(h)

        ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
        localized = np.where(ipr_vals >= ipr_threshold)[0]
        if localized.size == 0:
            raise ValueError("未找到局域态，无法构造 Case 3")
        loc_idx = localized[np.argmax(ipr_vals[localized])]
        psi_i = evecs[:, loc_idx]
        k_ext, _ = select_k_from_ratio(n_cells, r_ext, a=a)
        evals_k, evecs_k = bloch_eigensystem(k_ext, t0=t0, delta_t=delta_t, a=a)
        band_idx = int(ext_band)
        if band_idx not in (0, 1):
            raise ValueError("ext_band 必须为 0（价带）或 1（导带）")
        psi_j = bloch_state(n_cells, k_ext, evecs_k[:, band_idx], a=a)
        delta_e = float(evals_k[band_idx] - evals[loc_idx])

        q_vals = q_grid(n_cells, a=a)
        omegas, evecs_q = diatomic_modes_grid(q_vals, k_spring, mass_a, mass_b, a=a)
        g_vals = g_ssh_diatomic_grid(
            psi_i,
            psi_j,
            n_cells,
            q_vals,
            evecs_q,
            alpha,
            a=a,
            mass_a=mass_a,
            mass_b=mass_b,
        )
        qdot_var = qdot_variance(omegas, temperature, mode=mode)
        d2 = mean_square_nac(g_vals, delta_e, qdot_var)

        d2_vals.append(d2)
        delta_e_vals.append(delta_e)

    return CaseResult(
        n_vals=np.array(n_vals, dtype=int),
        d2_vals=np.array(d2_vals, dtype=float),
        delta_e_vals=np.array(delta_e_vals, dtype=float),
    )
