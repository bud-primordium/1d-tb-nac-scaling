"""诊断 SSH 版本的数值问题。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np

from src.diagnostics import ipr, top_ipr_indices
from src.electron_phonon import g_ssh_diatomic, g_ssh_diatomic_grid
from src.experiments_ssh import select_gap_k
from src.lattice import q_grid
from src.phonon_diatomic import diatomic_modes, diatomic_modes_grid
from src.ssh_electron import (
    bloch_eigensystem,
    bloch_state,
    build_hamiltonian,
    build_square_well,
    diagonalize,
)


def diagnose_case1():
    """诊断 Case 1 的 g 值。"""
    print("=" * 60)
    print("Case 1 诊断: VBM-CBM 耦合")
    print("=" * 60)

    t0, delta_t, alpha = 1.0, 0.2, 0.5
    k_spring, mass_a, mass_b, a = 1.0, 1.0, 1.0, 1.0

    for n_cells in [20, 40, 80, 160]:
        k_gap = select_gap_k(n_cells, t0=t0, delta_t=delta_t, a=a)
        evals_k, evecs_k = bloch_eigensystem(k_gap, t0=t0, delta_t=delta_t, a=a)
        psi_v = bloch_state(n_cells, k_gap, evecs_k[:, 0], a=a)
        psi_c = bloch_state(n_cells, k_gap, evecs_k[:, 1], a=a)

        # 检查波函数归一化
        norm_v = np.sum(np.abs(psi_v)**2)
        norm_c = np.sum(np.abs(psi_c)**2)
        overlap = np.abs(np.vdot(psi_v, psi_c))

        # q=0 光学模
        omega_q, evec_q = diatomic_modes(0.0, k_spring, mass_a, mass_b, a=a)
        mode_idx = int(np.argmax(omega_q))

        g_val = g_ssh_diatomic(
            psi_v, psi_c, n_cells, q=0.0, evec=evec_q[:, mode_idx],
            alpha=alpha, a=a, mass_a=mass_a, mass_b=mass_b
        )

        print(f"\nN = {n_cells}:")
        print(f"  k_gap = {k_gap:.6f}")
        print(f"  |psi_v|^2 = {norm_v:.10f}, |psi_c|^2 = {norm_c:.10f}")
        print(f"  <v|c> = {overlap:.2e}")
        print(f"  omega_optical = {omega_q[mode_idx]:.6f}")
        print(f"  evec_optical = {evec_q[:, mode_idx]}")
        print(f"  |g(q=0)|^2 = {np.abs(g_val)**2:.6e}")
        print(f"  |g(q=0)|^2 * N = {np.abs(g_val)**2 * n_cells:.6e}")


def diagnose_case2():
    """诊断 Case 2 的 g 值 - 这个应该是问题所在。"""
    print("\n" + "=" * 60)
    print("Case 2 诊断: Loc-Loc 耦合")
    print("=" * 60)

    t0, delta_t, alpha = 1.0, 0.2, 0.5
    k_spring, mass_a, mass_b, a = 1.0, 1.0, 1.0, 1.0
    well_width, well_depth = 5, -1.5
    ipr_threshold = 0.05

    for n_cells in [20, 40, 80, 160]:
        onsite = build_square_well(n_cells, center=n_cells//2, width=well_width, depth=well_depth)
        h = build_hamiltonian(n_cells, t0=t0, delta_t=delta_t, onsite=onsite, pbc=True)
        evals, evecs = diagonalize(h)

        ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
        localized = np.where(ipr_vals >= ipr_threshold)[0]

        print(f"\nN = {n_cells}:")
        print(f"  找到 {len(localized)} 个局域态 (IPR >= {ipr_threshold})")

        if len(localized) >= 2:
            local_ipr = ipr_vals[localized]
            order = np.argsort(local_ipr)[::-1]
            idx1, idx2 = localized[order[:2]]
            print(f"  选中态: idx1={idx1} (IPR={ipr_vals[idx1]:.4f}), idx2={idx2} (IPR={ipr_vals[idx2]:.4f})")
        else:
            idx1, idx2 = top_ipr_indices(evecs, count=2)
            print(f"  警告: 局域态不足2个，使用top IPR: idx1={idx1}, idx2={idx2}")

        psi_i = evecs[:, idx1]
        psi_j = evecs[:, idx2]

        # 检查波函数归一化
        norm_i = np.sum(np.abs(psi_i)**2)
        norm_j = np.sum(np.abs(psi_j)**2)
        overlap = np.abs(np.vdot(psi_i, psi_j))

        print(f"  |psi_i|^2 = {norm_i:.10f}, |psi_j|^2 = {norm_j:.10f}")
        print(f"  <i|j> = {overlap:.2e}")

        # 计算所有 q 模式的 g
        q_vals = q_grid(n_cells, a=a)
        omegas, evecs_q = diatomic_modes_grid(q_vals, k_spring, mass_a, mass_b, a=a)
        g_vals = g_ssh_diatomic_grid(
            psi_i, psi_j, n_cells, q_vals, evecs_q,
            alpha, a=a, mass_a=mass_a, mass_b=mass_b
        )

        g_flat = np.abs(g_vals).ravel()
        print(f"  |g|_max = {np.max(g_flat):.6e}")
        print(f"  |g|_mean = {np.mean(g_flat):.6e}")
        print(f"  sum |g|^2 = {np.sum(np.abs(g_vals)**2):.6e}")

        # 检查 q=0 模式
        g_q0 = g_vals[0, :]  # q=0 的两个模式
        print(f"  |g(q=0, acoustic)|^2 = {np.abs(g_q0[0])**2:.6e}")
        print(f"  |g(q=0, optical)|^2 = {np.abs(g_q0[1])**2:.6e}")

        # 检查 qdot_var 和 d2
        from src.nac import mean_square_nac, qdot_variance
        delta_e = float(evals[idx2] - evals[idx1])
        qdot_var = qdot_variance(omegas, temperature=300.0, mode="classical")
        d2 = mean_square_nac(g_vals, delta_e, qdot_var)

        print(f"  delta_E = {delta_e:.6f}")
        print(f"  <d^2> = {d2:.6e}")


def diagnose_g_matrix_element():
    """深入诊断 g 矩阵元的计算。"""
    print("\n" + "=" * 60)
    print("g 矩阵元计算诊断")
    print("=" * 60)

    from src.electron_phonon import dh_dq_ssh_diatomic

    t0, delta_t, alpha = 1.0, 0.2, 0.5
    k_spring, mass_a, mass_b, a = 1.0, 1.0, 1.0, 1.0
    n_cells = 40

    # Case 2 的局域态
    well_width, well_depth = 5, -1.5
    onsite = build_square_well(n_cells, center=n_cells//2, width=well_width, depth=well_depth)
    h = build_hamiltonian(n_cells, t0=t0, delta_t=delta_t, onsite=onsite, pbc=True)
    evals, evecs = diagonalize(h)

    ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
    localized = np.where(ipr_vals >= 0.05)[0]
    local_ipr = ipr_vals[localized]
    order = np.argsort(local_ipr)[::-1]
    idx1, idx2 = localized[order[:2]]

    psi_i = evecs[:, idx1]
    psi_j = evecs[:, idx2]

    print(f"psi_i shape: {psi_i.shape}, psi_j shape: {psi_j.shape}")
    print(f"psi_i 非零元素数: {np.sum(np.abs(psi_i) > 1e-10)}")
    print(f"psi_j 非零元素数: {np.sum(np.abs(psi_j) > 1e-10)}")

    # 手动计算 dH/dQ
    q = 0.0
    omega_q, evec_q = diatomic_modes(q, k_spring, mass_a, mass_b, a=a)
    mode_idx = 1  # 光学模

    dh = dh_dq_ssh_diatomic(n_cells, q, evec_q[:, mode_idx], alpha, a, mass_a, mass_b)

    print(f"\ndH/dQ 矩阵:")
    print(f"  shape: {dh.shape}")
    print(f"  |dH/dQ|_max = {np.max(np.abs(dh)):.6e}")
    print(f"  |dH/dQ|_Frobenius = {np.linalg.norm(dh):.6e}")
    print(f"  非零元素数: {np.sum(np.abs(dh) > 1e-15)}")

    # 分步计算 g = <i|dH/dQ|j>
    dh_psi_j = dh @ psi_j
    print(f"\ndH/dQ @ psi_j:")
    print(f"  |dH/dQ @ psi_j|_max = {np.max(np.abs(dh_psi_j)):.6e}")
    print(f"  非零元素数: {np.sum(np.abs(dh_psi_j) > 1e-15)}")

    g_val = np.vdot(psi_i, dh_psi_j)
    print(f"\ng = <i|dH/dQ|j> = {g_val}")
    print(f"|g|^2 = {np.abs(g_val)**2:.6e}")

    # 检查 psi_i 和 dH @ psi_j 的重叠区域
    psi_i_nonzero = np.abs(psi_i) > 1e-10
    dh_psi_j_nonzero = np.abs(dh_psi_j) > 1e-10
    overlap_region = psi_i_nonzero & dh_psi_j_nonzero
    print(f"\n重叠分析:")
    print(f"  psi_i 非零区域: {np.where(psi_i_nonzero)[0][:10]}...")
    print(f"  dH@psi_j 非零区域: {np.where(dh_psi_j_nonzero)[0][:10]}...")
    print(f"  重叠元素数: {np.sum(overlap_region)}")


def diagnose_simple_case2():
    """对比检查简化版 Case 2。"""
    print("\n" + "=" * 60)
    print("简化版 Case 2 对比诊断")
    print("=" * 60)

    from src.diagnostics import select_k_from_ratio
    from src.electron_phonon import g_monatomic, g_monatomic_grid
    from src.tb_electron_1band import (
        build_hamiltonian as simple_build_hamiltonian,
        build_square_well as simple_build_square_well,
        diagonalize as simple_diagonalize,
    )

    t0, alpha, a, mass = 1.0, 0.5, 1.0, 1.0
    well_width, well_depth = 5, -1.5
    ipr_threshold = 0.05

    for n_cells in [40, 80, 160]:
        onsite = simple_build_square_well(n_cells, center=n_cells//2, width=well_width, depth=well_depth)
        h = simple_build_hamiltonian(n_cells, t0=t0, onsite=onsite, pbc=True)
        evals, evecs = simple_diagonalize(h)

        ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])
        localized = np.where(ipr_vals >= ipr_threshold)[0]

        if len(localized) >= 2:
            local_ipr = ipr_vals[localized]
            order = np.argsort(local_ipr)[::-1]
            idx1, idx2 = localized[order[:2]]
        else:
            idx1, idx2 = top_ipr_indices(evecs, count=2)

        psi_i = evecs[:, idx1]
        psi_j = evecs[:, idx2]

        q_vals = np.array([2*np.pi*m/(n_cells*a) for m in range(n_cells)])
        g_vals = g_monatomic_grid(psi_i, psi_j, n_cells, q_vals, alpha, a=a, mass=mass)

        print(f"\nN = {n_cells} (简化版):")
        print(f"  |g|_max = {np.max(np.abs(g_vals)):.6e}")
        print(f"  sum |g|^2 = {np.sum(np.abs(g_vals)**2):.6e}")

        from src.phonon_1atom import dispersion_monatomic
        from src.nac import mean_square_nac, qdot_variance
        omegas = np.array([dispersion_monatomic(q, k_spring=1.0, mass=mass) for q in q_vals])
        delta_e = float(evals[idx2] - evals[idx1])
        qdot_var = qdot_variance(omegas, temperature=300.0, mode="classical")
        d2 = mean_square_nac(g_vals[:, np.newaxis], delta_e, qdot_var[:, np.newaxis])

        print(f"  <d^2> = {d2:.6e}")


if __name__ == "__main__":
    diagnose_case1()
    diagnose_case2()
    diagnose_g_matrix_element()
    diagnose_simple_case2()
