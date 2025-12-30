"""散射态选择的单元测试。"""

from __future__ import annotations

import numpy as np

from src.scattering_state import select_scattering_state


def test_select_scattering_state_degenerate_projection() -> None:
    """简并子空间内应重构出与参考态一致的线性组合。"""
    evals = np.array([0.0, 0.0])
    evecs = np.eye(2, dtype=complex)
    psi_ref = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)

    result = select_scattering_state(evals, evecs, psi_ref, energy_tol=1e-8)

    # 选中的态应与参考态重合（差一个全局相位已被对齐）
    assert np.isclose(result.overlap, 1.0, atol=1e-12)
    assert np.allclose(result.state, psi_ref, atol=1e-12)

