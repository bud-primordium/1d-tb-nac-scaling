"""简化版实验：单带单原子链。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .diagnostics import find_localized_states, ipr, select_bloch_states
from .electron_phonon import g_monatomic, g_monatomic_grid
from .lattice import k_grid, q_grid
from .nac import mean_square_nac, qdot_variance
from .phonon_1atom import dispersion_monatomic
from .tb_electron_1band import (
    bloch_state,
    build_hamiltonian,
    build_square_well,
    diagonalize,
    dispersion,
)


@dataclass
class CaseResult:
    """单个 Case 的结果容器。"""

    n_vals: np.ndarray
    d2_vals: np.ndarray
    delta_e_vals: np.ndarray


def folded_q_branches(m: int, a: float = 1.0, q_sc: float = 0.0) -> np.ndarray:
    """返回折叠到同一超胞 q_sc 的原胞 q 分支集合。"""
    g_sc = 2.0 * np.pi / (m * a)
    q_vals = q_sc + g_sc * np.arange(m, dtype=float)
    return q_vals


def case1_scaling(
    n_vals: List[int],
    r1: float,
    r2: float,
    t0: float,
    alpha: float,
    k_spring: float,
    temperature: float,
    a: float = 1.0,
    mass: float = 1.0,
    mode: str = "classical",
) -> CaseResult:
    """Case 1′：延展-延展的标度律。"""
    d2_vals = []
    delta_e_vals = []
    for n_cells in n_vals:
        k1, k2, _, _ = select_bloch_states(n_cells, r1, r2, a=a)
        psi_i = bloch_state(n_cells, k1, a=a)
        psi_j = bloch_state(n_cells, k2, a=a)
        q0 = k2 - k1

        g_q0 = g_monatomic(psi_i, psi_j, n_cells, q0, alpha, a=a, mass=mass)
        omega_q0 = dispersion_monatomic(np.array([q0]), k_spring, mass)[0]
        qdot_var = qdot_variance(np.array([omega_q0]), temperature, mode=mode)
        delta_e = dispersion(k2, t0, a=a) - dispersion(k1, t0, a=a)
        d2 = mean_square_nac(np.array([g_q0]), delta_e, qdot_var)

        d2_vals.append(d2)
        delta_e_vals.append(delta_e)

    return CaseResult(
        n_vals=np.array(n_vals, dtype=int),
        d2_vals=np.array(d2_vals, dtype=float),
        delta_e_vals=np.array(delta_e_vals, dtype=float),
    )


def folded_phonon_demo(
    n_cells: int,
    r1: float,
    r2: float,
    m_supercell: int,
    t0: float,
    alpha: float,
    a: float = 1.0,
    mass: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """折叠声子验证：返回折叠分支上的 g(q) 分布。"""
    k1, k2, _, _ = select_bloch_states(n_cells, r1, r2, a=a)
    psi_i = bloch_state(n_cells, k1, a=a)
    psi_j = bloch_state(n_cells, k2, a=a)
    q0 = k2 - k1

    q_branches = folded_q_branches(m_supercell, a=a, q_sc=0.0)
    g_vals = np.zeros_like(q_branches, dtype=complex)
    for idx, q in enumerate(q_branches):
        g_vals[idx] = g_monatomic(
            psi_i=psi_i,
            psi_j=psi_j,
            n_cells=n_cells,
            q=q,
            alpha=alpha,
            a=a,
            mass=mass,
        )
    return q_branches, g_vals


def case2_scaling(
    n_vals: List[int],
    t0: float,
    alpha: float,
    k_spring: float,
    temperature: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    a: float = 1.0,
    mass: float = 1.0,
    mode: str = "classical",
) -> CaseResult:
    """Case 2：局域-局域的标度律。"""
    d2_vals = []
    delta_e_vals = []
    for n_cells in n_vals:
        onsite = build_square_well(n_cells, center=n_cells // 2, width=well_width, depth=well_depth)
        h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
        evals, evecs = diagonalize(h)

        localized = find_localized_states(evecs, ipr_threshold)
        if localized.size < 2:
            raise ValueError("局域态数量不足，无法构造 Case 2")
        ipr_vals = np.array([ipr(evecs[:, i]) for i in localized])
        order = np.argsort(ipr_vals)[::-1]
        idx1, idx2 = localized[order[:2]]
        psi_i = evecs[:, idx1]
        psi_j = evecs[:, idx2]
        delta_e = float(evals[idx2] - evals[idx1])

        q_vals = q_grid(n_cells, a=a)
        g_vals = g_monatomic_grid(psi_i, psi_j, n_cells, q_vals, alpha, a=a, mass=mass)
        omegas = dispersion_monatomic(q_vals, k_spring, mass)
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
    alpha: float,
    k_spring: float,
    temperature: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    a: float = 1.0,
    mass: float = 1.0,
    mode: str = "classical",
) -> CaseResult:
    """Case 3：局域-延展的标度律。"""
    d2_vals = []
    delta_e_vals = []
    for n_cells in n_vals:
        onsite = build_square_well(n_cells, center=n_cells // 2, width=well_width, depth=well_depth)
        h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
        evals, evecs = diagonalize(h)

        ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
        localized_indices = np.where(ipr_vals >= ipr_threshold)[0]
        if localized_indices.size == 0:
            raise ValueError("未找到局域态，无法构造 Case 3")
        loc_idx = localized_indices[np.argmax(ipr_vals[localized_indices])]
        ext_idx = int(np.argmin(ipr_vals))

        psi_i = evecs[:, loc_idx]
        psi_j = evecs[:, ext_idx]
        delta_e = float(evals[ext_idx] - evals[loc_idx])

        q_vals = q_grid(n_cells, a=a)
        g_vals = g_monatomic_grid(psi_i, psi_j, n_cells, q_vals, alpha, a=a, mass=mass)
        omegas = dispersion_monatomic(q_vals, k_spring, mass)
        qdot_var = qdot_variance(omegas, temperature, mode=mode)
        d2 = mean_square_nac(g_vals, delta_e, qdot_var)

        d2_vals.append(d2)
        delta_e_vals.append(delta_e)

    return CaseResult(
        n_vals=np.array(n_vals, dtype=int),
        d2_vals=np.array(d2_vals, dtype=float),
        delta_e_vals=np.array(delta_e_vals, dtype=float),
    )
