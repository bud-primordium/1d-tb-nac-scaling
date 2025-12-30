"""简化版实验：单带单原子链。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .diagnostics import (
    find_localized_states,
    ipr,
    select_bloch_states,
    select_k_from_ratio,
    top_ipr_indices,
)
from .electron_phonon import g_monatomic, g_monatomic_grid
from .lattice import k_grid, q_grid
from .nac import mean_square_nac, qdot_variance
from .phonon_1atom import dispersion_monatomic
from .scattering_state import select_scattering_state
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


@dataclass
class BlochCharacterResult:
    """Bloch 特征随尺寸的诊断结果。"""

    n_vals: np.ndarray
    overlaps: np.ndarray
    indices: np.ndarray


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
        k1, k2, m1, m2 = select_bloch_states(n_cells, r1, r2, a=a)
        psi_i = bloch_state(n_cells, k1, a=a)
        psi_j = bloch_state(n_cells, k2, a=a)
        q_index = (m1 - m2) % n_cells
        q0 = 2.0 * np.pi * q_index / (n_cells * a)

        g_q0 = g_monatomic(psi_i, psi_j, n_cells, q0, alpha, a=a, mass=mass)
        omega_q0 = dispersion_monatomic(np.array([q0]), k_spring, mass)[0]
        qdot_var = qdot_variance(np.array([omega_q0]), temperature, mode=mode)
        delta_e = dispersion(k2, t0, a=a) - dispersion(k1, t0, a=a)
        d2 = mean_square_nac(np.array([g_q0]), delta_e, qdot_var, n_cells=n_cells)

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
    _ = t0
    k1, k2, m1, m2 = select_bloch_states(n_cells, r1, r2, a=a)
    psi_i = bloch_state(n_cells, k1, a=a)
    psi_j = bloch_state(n_cells, k2, a=a)
    q_index = (m1 - m2) % n_cells
    q0 = 2.0 * np.pi * q_index / (n_cells * a)

    q_branches = folded_q_branches(m_supercell, a=a, q_sc=0.0)
    step = n_cells // m_supercell
    if q_index % step != 0:
        raise ValueError("折叠分支未命中 q0，请调整 m_supercell 或 k 点选择")
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
        if localized.size >= 2:
            ipr_vals = np.array([ipr(evecs[:, i]) for i in localized])
            order = np.argsort(ipr_vals)[::-1]
            idx1, idx2 = localized[order[:2]]
        else:
            idx1, idx2 = top_ipr_indices(evecs, count=2)
        psi_i = evecs[:, idx1]
        psi_j = evecs[:, idx2]
        delta_e = float(evals[idx2] - evals[idx1])

        q_vals = q_grid(n_cells, a=a)
        g_vals = g_monatomic_grid(psi_i, psi_j, n_cells, q_vals, alpha, a=a, mass=mass)
        omegas = dispersion_monatomic(q_vals, k_spring, mass)
        qdot_var = qdot_variance(omegas, temperature, mode=mode)
        d2 = mean_square_nac(g_vals, delta_e, qdot_var, n_cells=n_cells)

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
    r_ext: float,
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
        psi_i = evecs[:, loc_idx]
        k_ext, _ = select_k_from_ratio(n_cells, r_ext, a=a)
        psi_j = bloch_state(n_cells, k_ext, a=a)
        delta_e = float(dispersion(k_ext, t0, a=a) - evals[loc_idx])

        q_vals = q_grid(n_cells, a=a)
        g_vals = g_monatomic_grid(psi_i, psi_j, n_cells, q_vals, alpha, a=a, mass=mass)
        omegas = dispersion_monatomic(q_vals, k_spring, mass)
        qdot_var = qdot_variance(omegas, temperature, mode=mode)
        d2 = mean_square_nac(g_vals, delta_e, qdot_var, n_cells=n_cells)

        d2_vals.append(d2)
        delta_e_vals.append(delta_e)

    return CaseResult(
        n_vals=np.array(n_vals, dtype=int),
        d2_vals=np.array(d2_vals, dtype=float),
        delta_e_vals=np.array(delta_e_vals, dtype=float),
    )


def case3_scaling_strict(
    n_vals: List[int],
    t0: float,
    alpha: float,
    k_spring: float,
    temperature: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    energy_window: float | None = None,
    a: float = 1.0,
    mass: float = 1.0,
    mode: str = "classical",
) -> CaseResult:
    """Case 3（严格版）：局域-延展（延展态取含缺陷散射态的严格本征态）。"""
    d2_vals = []
    delta_e_vals = []
    for n_cells in n_vals:
        onsite = build_square_well(n_cells, center=n_cells // 2, width=well_width, depth=well_depth)
        h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
        evals, evecs = diagonalize(h)

        ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
        localized_indices = np.where(ipr_vals >= ipr_threshold)[0]
        if localized_indices.size == 0:
            raise ValueError("未找到局域态，无法构造 Case 3（严格版）")
        loc_idx = localized_indices[np.argmax(ipr_vals[localized_indices])]
        psi_loc = evecs[:, loc_idx]

        k_ext, _ = select_k_from_ratio(n_cells, r_ext, a=a)
        psi_ref = bloch_state(n_cells, k_ext, a=a)

        allowed = np.where(ipr_vals < ipr_threshold)[0]
        selection = select_scattering_state(
            evals=evals,
            evecs=evecs,
            psi_ref=psi_ref,
            energy_ref=dispersion(k_ext, t0, a=a),
            energy_window=energy_window,
            allowed_indices=allowed,
        )
        psi_ext = selection.state
        delta_e = float(evals[selection.idx] - evals[loc_idx])

        q_vals = q_grid(n_cells, a=a)
        g_vals = g_monatomic_grid(psi_loc, psi_ext, n_cells, q_vals, alpha, a=a, mass=mass)
        omegas = dispersion_monatomic(q_vals, k_spring, mass)
        qdot_var = qdot_variance(omegas, temperature, mode=mode)
        d2 = mean_square_nac(g_vals, delta_e, qdot_var, n_cells=n_cells)

        d2_vals.append(d2)
        delta_e_vals.append(delta_e)

    return CaseResult(
        n_vals=np.array(n_vals, dtype=int),
        d2_vals=np.array(d2_vals, dtype=float),
        delta_e_vals=np.array(delta_e_vals, dtype=float),
    )


def bloch_character_vs_N(
    n_vals: List[int],
    t0: float,
    well_width: int,
    well_depth: float,
    ipr_threshold: float,
    r_ext: float,
    a: float = 1.0,
) -> BlochCharacterResult:
    """返回含缺陷体系中“最像 Bloch 参考态”的本征态重叠随 N 的变化。"""
    overlaps = []
    indices = []
    for n_cells in n_vals:
        onsite = build_square_well(n_cells, center=n_cells // 2, width=well_width, depth=well_depth)
        h = build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
        _, evecs = diagonalize(h)

        ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
        allowed = np.where(ipr_vals < ipr_threshold)[0]

        k_ext, _ = select_k_from_ratio(n_cells, r_ext, a=a)
        psi_ref = bloch_state(n_cells, k_ext, a=a)
        coeffs = evecs.conj().T @ psi_ref
        weights = np.abs(coeffs)
        weights = np.where(np.isin(np.arange(weights.size), allowed), weights, 0.0)
        idx = int(np.argmax(weights))
        overlaps.append(float(weights[idx]))
        indices.append(idx)

    return BlochCharacterResult(
        n_vals=np.array(n_vals, dtype=int),
        overlaps=np.array(overlaps, dtype=float),
        indices=np.array(indices, dtype=int),
    )
