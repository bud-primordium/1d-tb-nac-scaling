"""生成出版级图表 - NAC 尺寸效应验证。

此脚本生成用于报告的所有图表，采用统一的学术风格。
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
from matplotlib import font_manager
from scipy import stats

# 导入计算模块
from src.diagnostics import ipr, select_k_from_ratio, select_bloch_states, top_ipr_indices
from src.electron_phonon import g_monatomic, g_monatomic_grid, g_ssh_diatomic, g_ssh_diatomic_grid
from src.experiments_simple import case1_scaling as simple_case1, case2_scaling as simple_case2, case3_scaling as simple_case3
from src.experiments_ssh import case1_scaling as ssh_case1, case2_scaling as ssh_case2, case3_scaling as ssh_case3, select_gap_k
from src.lattice import k_grid, q_grid
from src.phonon_1atom import dispersion_monatomic
from src.phonon_diatomic import diatomic_modes, diatomic_modes_grid
from src.ssh_electron import bloch_eigensystem, bloch_state, build_hamiltonian, build_square_well, diagonalize
from src.tb_electron_1band import (
    bloch_state as simple_bloch_state,
    build_hamiltonian as simple_build_hamiltonian,
    build_square_well as simple_build_square_well,
    diagonalize as simple_diagonalize,
    dispersion as simple_dispersion,
)


# =============================================================================
# 样式设置
# =============================================================================

# 配色方案
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
        # 字体 - 使用更清晰的字体
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'PingFang SC', 'Heiti SC'],
        'font.size': 10,
        'mathtext.fontset': 'stix',  # 使用更好看的数学字体
        # 坐标轴
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'axes.labelpad': 4,
        'axes.unicode_minus': False,
        'axes.formatter.useoffset': False,  # 禁用自动offset
        'axes.formatter.use_mathtext': True,
        # 刻度
        'xtick.major.size': 4,
        'xtick.major.width': 1.0,
        'xtick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.major.size': 4,
        'ytick.major.width': 1.0,
        'ytick.labelsize': 10,
        'ytick.direction': 'in',
        # 图例
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        # 线条
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        # 网格
        'axes.grid': False,
        # 图像
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def fit_power_law(n_vals, y_vals, min_n=40):
    """拟合幂律 y = A * N^(-beta)，返回 beta 和不确定度。"""
    mask = np.array(n_vals) >= min_n
    log_n = np.log(np.array(n_vals)[mask])
    log_y = np.log(np.array(y_vals)[mask])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_y)
    return -slope, std_err, r_value**2


def ensure_output_dir():
    """确保输出目录存在。"""
    output_dir = ROOT / "results" / "publication_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# 参数配置
# =============================================================================

# 简化版参数
SIMPLE_PARAMS = {
    'n_vals': [20, 40, 60, 80, 120, 160, 240, 320, 480, 640],
    'r1': 1/10,
    'r2': 3/10,
    't0': 1.0,
    'alpha': 0.5,
    'k_spring': 1.0,
    'temperature': 300.0,
    'a': 1.0,
    'mass': 1.0,
    'well_width': 5,
    'well_depth': -1.5,
    'ipr_threshold': 0.05,
}

# SSH 参数
SSH_PARAMS = {
    'n_vals': [20, 40, 60, 80, 120, 160, 240, 320, 480, 640],
    't0': 1.0,
    'delta_t': 0.2,
    'alpha': 0.5,
    'k_spring': 1.0,
    'temperature': 300.0,
    'a': 1.0,
    'mass_a': 1.0,
    'mass_b': 1.0,
    'well_width': 5,
    'well_depth': -1.5,
    'ipr_threshold': 0.05,
    'r_ext': 1/10,
}


# =============================================================================
# 图表生成函数
# =============================================================================

def generate_figure1_model_schematic(output_dir: Path):
    """Figure 1: 模型示意图。"""
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

    # (a) 单带单原子链
    ax = axes[0]
    n_sites = 7
    x = np.arange(n_sites)
    ax.scatter(x, np.zeros(n_sites), s=200, c=COLORS['case1'], zorder=3)
    for i in range(n_sites - 1):
        ax.plot([x[i], x[i+1]], [0, 0], 'k-', lw=2, zorder=2)
        ax.text((x[i] + x[i+1])/2, 0.15, r'$t_0$', ha='center', fontsize=9)
    ax.set_xlim(-0.5, n_sites - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Monatomic chain', fontsize=11)

    # (b) SSH 双原子链
    ax = axes[1]
    n_cells = 4
    x_a = np.arange(n_cells) * 2
    x_b = np.arange(n_cells) * 2 + 0.6
    ax.scatter(x_a, np.zeros(n_cells), s=200, c=COLORS['case1'], marker='o', zorder=3, label='A')
    ax.scatter(x_b, np.zeros(n_cells), s=150, c=COLORS['case2'], marker='s', zorder=3, label='B')
    for i in range(n_cells):
        # 胞内键 (v)
        ax.plot([x_a[i], x_b[i]], [0, 0], 'k-', lw=3, zorder=2)
        if i < n_cells - 1:
            # 胞间键 (w)
            ax.plot([x_b[i], x_a[i+1]], [0, 0], 'k-', lw=1.5, zorder=2)
    ax.text((x_a[1] + x_b[1])/2, 0.2, r'$v$', ha='center', fontsize=9)
    ax.text((x_b[0] + x_a[1])/2, 0.2, r'$w$', ha='center', fontsize=9)
    ax.set_xlim(-0.5, x_b[-1] + 0.5)
    ax.set_ylim(-0.5, 0.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) SSH diatomic chain', fontsize=11)

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_model_schematic.png")
    plt.close(fig)
    print("  Figure 1: 模型示意图 - 完成")


def generate_figure2_wavefunctions(output_dir: Path):
    """Figure 2: 三类态的波函数与IPR。"""
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

    n_cells = 80
    p = SIMPLE_PARAMS

    # Case 1: 延展-延展
    k1, k2, _, _ = select_bloch_states(n_cells, p['r1'], p['r2'], a=p['a'])
    psi_ext1 = simple_bloch_state(n_cells, k1, a=p['a'])
    psi_ext2 = simple_bloch_state(n_cells, k2, a=p['a'])

    # Case 2 & 3: 需要缺陷
    onsite = simple_build_square_well(n_cells, center=n_cells//2, width=p['well_width'], depth=p['well_depth'])
    h = simple_build_hamiltonian(n_cells, t0=p['t0'], onsite=onsite, pbc=True)
    evals, evecs = simple_diagonalize(h)
    ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])

    # 找局域态
    localized_idx = np.where(ipr_vals >= p['ipr_threshold'])[0]
    if len(localized_idx) >= 2:
        order = np.argsort(ipr_vals[localized_idx])[::-1]
        loc_idx1 = localized_idx[order[0]]
        loc_idx2 = localized_idx[order[1]]
    else:
        loc_idx1, loc_idx2 = top_ipr_indices(evecs, count=2)

    psi_loc1 = evecs[:, loc_idx1]
    psi_loc2 = evecs[:, loc_idx2]

    # 上排: |ψ|²
    x = np.arange(n_cells)

    axes[0, 0].fill_between(x, 0, np.abs(psi_ext1)**2, alpha=0.7, color=COLORS['case1'])
    axes[0, 0].fill_between(x, 0, np.abs(psi_ext2)**2, alpha=0.5, color=COLORS['case1'], linestyle='--')
    axes[0, 0].set_ylabel(r'$|\psi|^2$')
    axes[0, 0].set_title('(a) Case 1: Extended', fontsize=10)
    axes[0, 0].set_xlim(0, n_cells)

    axes[0, 1].fill_between(x, 0, np.abs(psi_loc1)**2, alpha=0.7, color=COLORS['case2'], label='L1')
    axes[0, 1].fill_between(x, 0, np.abs(psi_loc2)**2, alpha=0.6, color=COLORS['highlight'], label='L2')
    axes[0, 1].set_title('(b) Case 2: Localized', fontsize=10)
    axes[0, 1].legend(fontsize=7, loc='upper right')
    axes[0, 1].set_xlim(0, n_cells)

    axes[0, 2].fill_between(x, 0, np.abs(psi_loc1)**2, alpha=0.7, color=COLORS['case2'], label='Loc')
    axes[0, 2].fill_between(x, 0, np.abs(psi_ext1)**2, alpha=0.5, color=COLORS['case3'], label='Ext')
    axes[0, 2].set_title('(c) Case 3: Mixed', fontsize=10)
    axes[0, 2].legend(fontsize=7, loc='upper right')
    axes[0, 2].set_xlim(0, n_cells)

    # 下排: IPR vs Energy
    for i, ax in enumerate(axes[1, :]):
        ax.scatter(evals, ipr_vals, s=15, alpha=0.6, c='gray')
        ax.set_xlabel('Energy $E$')
        ax.axhline(y=p['ipr_threshold'], color='r', linestyle='--', lw=0.8, alpha=0.7)
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1)

    axes[1, 0].scatter([simple_dispersion(k1, p['t0'])], [1/n_cells], s=50, c=COLORS['case1'], marker='*', zorder=5)
    axes[1, 0].scatter([simple_dispersion(k2, p['t0'])], [1/n_cells], s=50, c=COLORS['case1'], marker='*', zorder=5)
    axes[1, 0].set_ylabel('IPR')
    axes[1, 0].set_title('(d) IPR spectrum', fontsize=10)

    axes[1, 1].scatter([evals[loc_idx1], evals[loc_idx2]], [ipr_vals[loc_idx1], ipr_vals[loc_idx2]],
                       s=50, c=COLORS['case2'], marker='*', zorder=5)
    axes[1, 1].set_title('(e) IPR spectrum', fontsize=10)

    axes[1, 2].scatter([evals[loc_idx1]], [ipr_vals[loc_idx1]], s=50, c=COLORS['case2'], marker='*', zorder=5)
    axes[1, 2].scatter([simple_dispersion(k1, p['t0'])], [1/n_cells], s=50, c=COLORS['case3'], marker='*', zorder=5)
    axes[1, 2].set_title('(f) IPR spectrum', fontsize=10)

    for ax in axes[0, :]:
        ax.set_xlabel('Site $n$')

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_wavefunctions.png")
    plt.close(fig)
    print("  Figure 2: 波函数与IPR - 完成")


def generate_figure3_scaling_laws(output_dir: Path):
    """Figure 3: 标度律主图 (核心图)。"""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5))

    p = SIMPLE_PARAMS
    n_vals = p['n_vals']

    print("  计算简化版数据...")
    # 简化版数据
    result1 = simple_case1(
        n_vals=n_vals, r1=p['r1'], r2=p['r2'], t0=p['t0'], alpha=p['alpha'],
        k_spring=p['k_spring'], temperature=p['temperature'], a=p['a'], mass=p['mass']
    )
    result2 = simple_case2(
        n_vals=n_vals, t0=p['t0'], alpha=p['alpha'], k_spring=p['k_spring'],
        temperature=p['temperature'], well_width=p['well_width'], well_depth=p['well_depth'],
        ipr_threshold=p['ipr_threshold'], a=p['a'], mass=p['mass']
    )
    result3 = simple_case3(
        n_vals=n_vals, t0=p['t0'], alpha=p['alpha'], k_spring=p['k_spring'],
        temperature=p['temperature'], well_width=p['well_width'], well_depth=p['well_depth'],
        ipr_threshold=p['ipr_threshold'], r_ext=p['r1'], a=p['a'], mass=p['mass']
    )

    # (a) Case 1
    ax = axes[0, 0]
    beta1, err1, r2_1 = fit_power_law(result1.n_vals, result1.d2_vals)
    ax.scatter(result1.n_vals, result1.d2_vals, s=40, c=COLORS['case1'], zorder=3, label='Data')
    # 拟合线
    n_fit = np.linspace(20, 700, 100)
    A1 = result1.d2_vals[0] * result1.n_vals[0]**beta1
    ax.plot(n_fit, A1 * n_fit**(-beta1), '--', c=COLORS['case1'], lw=1.5, alpha=0.8, label=f'Fit: $\\beta={beta1:.2f}$')
    # 理论线
    ax.plot(n_fit, A1 * n_fit**(-1), ':', c=COLORS['theory'], lw=1.2, label=r'Theory: $N^{-1}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$\langle d^2 \rangle$')
    ax.set_title(f'(a) Case 1: Ext-Ext\n$\\beta = {beta1:.2f} \\pm {err1:.2f}$', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')

    # (b) Case 2 - 使用均值±标准差展示常数行为
    ax = axes[0, 1]
    mean_d2 = np.mean(result2.d2_vals)
    std_d2 = np.std(result2.d2_vals)
    rel_std = std_d2 / mean_d2   # 相对标准差 (无%)
    ax.scatter(result2.n_vals, result2.d2_vals, s=40, c=COLORS['case2'], zorder=3, label='Data')
    # 理论线 (常数) 和误差带
    ax.axhline(y=mean_d2, linestyle='-', c=COLORS['theory'], lw=1.5, label=f'Mean')
    ax.axhspan(mean_d2 - std_d2, mean_d2 + std_d2, alpha=0.2, color=COLORS['case2'], label=r'$\pm 1\sigma$')
    ax.set_xscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$\langle d^2 \rangle$')
    ax.set_title(f'(b) Case 2: Loc-Loc\n$\\langle d^2 \\rangle = {mean_d2:.3e}, \\sigma/\\mu = {rel_std:.2e}$', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')

    # (c) Case 3
    ax = axes[1, 0]
    beta3, err3, r2_3 = fit_power_law(result3.n_vals, result3.d2_vals)
    ax.scatter(result3.n_vals, result3.d2_vals, s=40, c=COLORS['case3'], zorder=3, label='Data')
    # 拟合线
    A3 = result3.d2_vals[0] * result3.n_vals[0]**beta3
    ax.plot(n_fit, A3 * n_fit**(-beta3), '--', c=COLORS['case3'], lw=1.5, alpha=0.8, label=f'Fit: $\\beta={beta3:.2f}$')
    # 理论线
    ax.plot(n_fit, A3 * n_fit**(-1), ':', c=COLORS['theory'], lw=1.2, label=r'Theory: $N^{-1}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$\langle d^2 \rangle$')
    ax.set_title(f'(c) Case 3: Loc-Ext\n$\\beta = {beta3:.2f} \\pm {err3:.2f}$', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')

    # (d) 汇总表格
    ax = axes[1, 1]
    ax.axis('off')

    # 使用文本绘制表格
    ax.text(0.5, 0.95, 'Scaling Law Summary', ha='center', va='top', fontsize=11, fontweight='bold', transform=ax.transAxes)

    # Case 2 使用相对标准差而非幂律拟合
    cell_text = [
        ['Case', 'Type', 'Expected', 'Result', 'Quality'],
        ['1', 'Ext-Ext', r'$\beta=1$', f'$\\beta={beta1:.2f}$', f'$R^2={r2_1:.3f}$'],
        ['2', 'Loc-Loc', 'const', f'$\\sigma/\\mu={rel_std:.2e}$', 'Stable' if rel_std < 0.05 else 'Varies'],
        ['3', 'Loc-Ext', r'$\beta=1$', f'$\\beta={beta3:.2f}$', f'$R^2={r2_3:.3f}$'],
    ]

    table = ax.table(cellText=cell_text[1:], colLabels=cell_text[0],
                     loc='center', cellLoc='center',
                     colWidths=[0.15, 0.20, 0.24, 0.40, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # 标记通过/失败 - Case 2 用相对标准差判断
    case1_pass = abs(beta1 - 1.0) < 0.15
    case2_pass = rel_std < 0.05  # 相对标准差 < 5% 视为常数
    case3_pass = abs(beta3 - 1.0) < 0.15
    all_pass = case1_pass and case2_pass and case3_pass

    if all_pass:
        ax.text(0.5, 0.15, 'All cases PASSED', ha='center', va='bottom',
                fontsize=10, color='green', fontweight='bold', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.15, 'Some cases FAILED', ha='center', va='bottom',
                fontsize=10, color='red', fontweight='bold', transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_scaling_laws.png")
    plt.close(fig)
    print("  Figure 3: 标度律主图 - 完成")

    return {'case1': (beta1, err1), 'case2': (mean_d2, rel_std), 'case3': (beta3, err3)}


def generate_figure4_gq_distribution(output_dir: Path):
    """Figure 4: g(q) 分布对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    n_cells = 80
    p = SIMPLE_PARAMS

    # Case 1: 延展-延展 (只有一个 q 贡献)
    k1, k2, m1, m2 = select_bloch_states(n_cells, p['r1'], p['r2'], a=p['a'])
    psi_i = simple_bloch_state(n_cells, k1, a=p['a'])
    psi_j = simple_bloch_state(n_cells, k2, a=p['a'])
    q_vals = q_grid(n_cells, a=p['a'])
    g_case1 = g_monatomic_grid(psi_i, psi_j, n_cells, q_vals, p['alpha'], a=p['a'], mass=p['mass'])

    # Case 2 & 3
    onsite = simple_build_square_well(n_cells, center=n_cells//2, width=p['well_width'], depth=p['well_depth'])
    h = simple_build_hamiltonian(n_cells, t0=p['t0'], onsite=onsite, pbc=True)
    evals, evecs = simple_diagonalize(h)
    ipr_vals = np.array([ipr(evecs[:, i]) for i in range(evecs.shape[1])])

    localized_idx = np.where(ipr_vals >= p['ipr_threshold'])[0]
    if len(localized_idx) >= 2:
        order = np.argsort(ipr_vals[localized_idx])[::-1]
        loc_idx1 = localized_idx[order[0]]
        loc_idx2 = localized_idx[order[1]]
    else:
        loc_idx1, loc_idx2 = top_ipr_indices(evecs, count=2)

    psi_loc1 = evecs[:, loc_idx1]
    psi_loc2 = evecs[:, loc_idx2]

    g_case2 = g_monatomic_grid(psi_loc1, psi_loc2, n_cells, q_vals, p['alpha'], a=p['a'], mass=p['mass'])

    k_ext, _ = select_k_from_ratio(n_cells, p['r1'], a=p['a'])
    psi_ext = simple_bloch_state(n_cells, k_ext, a=p['a'])
    g_case3 = g_monatomic_grid(psi_loc1, psi_ext, n_cells, q_vals, p['alpha'], a=p['a'], mass=p['mass'])

    # 绘图
    axes[0].bar(q_vals, np.abs(g_case1)**2, width=0.08, color=COLORS['case1'], alpha=0.8)
    axes[0].set_xlabel('$q$')
    axes[0].set_ylabel(r'$|g(q)|^2$')
    axes[0].set_title(r'(a) Case 1: $\delta$-peak', fontsize=10)
    axes[0].set_xlim(-0.5, 2*np.pi + 0.5)

    axes[1].bar(q_vals, np.abs(g_case2)**2, width=0.08, color=COLORS['case2'], alpha=0.8)
    axes[1].set_xlabel('$q$')
    axes[1].set_title('(b) Case 2: Broad', fontsize=10)
    axes[1].set_xlim(-0.5, 2*np.pi + 0.5)

    axes[2].bar(q_vals, np.abs(g_case3)**2, width=0.08, color=COLORS['case3'], alpha=0.8)
    axes[2].set_xlabel('$q$')
    axes[2].set_title('(c) Case 3: Broad', fontsize=10)
    axes[2].set_xlim(-0.5, 2*np.pi + 0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_gq_distribution.png")
    plt.close(fig)
    print("  Figure 4: g(q)分布 - 完成")


def generate_figure5_folded_phonon(output_dir: Path):
    """Figure 5: 折叠声子验证。"""
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

    # (a) 简化版
    from src.experiments_simple import folded_phonon_demo as simple_folded
    p = SIMPLE_PARAMS
    n_cells = 40
    q_branches, g_vals = simple_folded(
        n_cells=n_cells, r1=p['r1'], r2=p['r2'], m_supercell=5,
        t0=p['t0'], alpha=p['alpha'], a=p['a'], mass=p['mass']
    )

    ax = axes[0]
    bars = ax.bar(np.arange(len(q_branches)), np.abs(g_vals), color=COLORS['case1'], alpha=0.8)
    # 高亮非零的
    max_idx = np.argmax(np.abs(g_vals))
    bars[max_idx].set_color(COLORS['highlight'])
    ax.set_xlabel('Folded branch index')
    ax.set_ylabel(r'$|g(q)|$')
    ax.set_title('(a) Simple: Only one nonzero', fontsize=10)
    ax.set_xticks(np.arange(len(q_branches)))

    # (b) SSH 版
    from src.experiments_ssh import folded_phonon_demo as ssh_folded
    ps = SSH_PARAMS
    g_ssh, _, _ = ssh_folded(
        n_cells=2, t0=ps['t0'], delta_t=ps['delta_t'], alpha=ps['alpha'],
        k_spring=ps['k_spring'], a=ps['a'], mass_a=ps['mass_a'], mass_b=ps['mass_b']
    )

    ax = axes[1]
    x_pos = [0, 1]
    colors = [COLORS['highlight'], COLORS['theory']]
    bars = ax.bar(x_pos, np.abs(g_ssh), color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Mode I\n(optical)', 'Mode II\n(folded)'])
    ax.set_ylabel(r'$|g|$')
    ax.set_title('(b) SSH: Mode I nonzero', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_folded_phonon.png")
    plt.close(fig)
    print("  Figure 5: 折叠声子验证 - 完成")


def generate_figure6_ssh_full(output_dir: Path):
    """Figure 6: SSH 完整版标度律。"""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    ps = SSH_PARAMS
    n_vals = ps['n_vals']

    print("  计算SSH版数据...")
    result1 = ssh_case1(
        n_vals=n_vals, t0=ps['t0'], delta_t=ps['delta_t'], alpha=ps['alpha'],
        k_spring=ps['k_spring'], temperature=ps['temperature'], a=ps['a'],
        mass_a=ps['mass_a'], mass_b=ps['mass_b']
    )
    result2 = ssh_case2(
        n_vals=n_vals, t0=ps['t0'], delta_t=ps['delta_t'], alpha=ps['alpha'],
        k_spring=ps['k_spring'], temperature=ps['temperature'],
        well_width=ps['well_width'], well_depth=ps['well_depth'],
        ipr_threshold=ps['ipr_threshold'], a=ps['a'],
        mass_a=ps['mass_a'], mass_b=ps['mass_b']
    )
    result3 = ssh_case3(
        n_vals=n_vals, t0=ps['t0'], delta_t=ps['delta_t'], alpha=ps['alpha'],
        k_spring=ps['k_spring'], temperature=ps['temperature'],
        well_width=ps['well_width'], well_depth=ps['well_depth'],
        ipr_threshold=ps['ipr_threshold'], r_ext=ps['r_ext'], ext_band=0, a=ps['a'],
        mass_a=ps['mass_a'], mass_b=ps['mass_b']
    )

    n_fit = np.linspace(20, 700, 100)

    # Case 1
    ax = axes[0]
    beta1, err1, _ = fit_power_law(result1.n_vals, result1.d2_vals)
    ax.scatter(result1.n_vals, result1.d2_vals, s=40, c=COLORS['case1'], zorder=3)
    A1 = result1.d2_vals[0] * result1.n_vals[0]**beta1
    ax.plot(n_fit, A1 * n_fit**(-beta1), '--', c=COLORS['case1'], lw=1.5, alpha=0.8)
    ax.plot(n_fit, A1 * n_fit**(-1), ':', c=COLORS['theory'], lw=1.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$\langle d^2 \rangle$')
    ax.set_title(f'(a) SSH Case 1: Ext-Ext\n$\\beta = {beta1:.2f}$', fontsize=10)

    # Case 2 - 使用均值±标准差展示常数行为
    ax = axes[1]
    mean_d2 = np.mean(result2.d2_vals)
    std_d2 = np.std(result2.d2_vals)
    rel_std = std_d2 / mean_d2 * 100  # 相对标准差 (%)
    ax.scatter(result2.n_vals, result2.d2_vals, s=40, c=COLORS['case2'], zorder=3)
    ax.axhline(y=mean_d2, linestyle='-', c=COLORS['theory'], lw=1.5)
    ax.axhspan(mean_d2 - std_d2, mean_d2 + std_d2, alpha=0.2, color=COLORS['case2'])
    ax.set_xscale('log')
    ax.set_yscale('log')  # 使用对数坐标避免 offset 问题
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$\langle d^2 \rangle$')
    ax.set_title(f'(b) SSH Case 2: Loc-Loc\n$\\sigma/\\mu = {rel_std:.1f}\\%$', fontsize=10)
    # 设置 y 轴范围让常数线可见
    ax.set_ylim(mean_d2 * 0.5, mean_d2 * 2)

    # Case 3
    ax = axes[2]
    beta3, err3, _ = fit_power_law(result3.n_vals, result3.d2_vals)
    ax.scatter(result3.n_vals, result3.d2_vals, s=40, c=COLORS['case3'], zorder=3)
    A3 = result3.d2_vals[0] * result3.n_vals[0]**beta3
    ax.plot(n_fit, A3 * n_fit**(-beta3), '--', c=COLORS['case3'], lw=1.5, alpha=0.8)
    ax.plot(n_fit, A3 * n_fit**(-1), ':', c=COLORS['theory'], lw=1.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    ax.set_title(f'(c) SSH Case 3: Loc-Ext\n$\\beta = {beta3:.2f}$', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "fig6_ssh_scaling.png")
    plt.close(fig)
    print("  Figure 6: SSH标度律 - 完成")

    return {'case1': (beta1, err1), 'case2': (mean_d2, rel_std), 'case3': (beta3, err3)}


def generate_figure7_diagnostics(output_dir: Path):
    """Figure 7: 诊断图 (附录)。"""
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

    p = SIMPLE_PARAMS
    n_vals = p['n_vals']

    # 计算诊断数据
    delta_e_case1 = []
    gq0_case1 = []

    for n_cells in n_vals:
        k1, k2, m1, m2 = select_bloch_states(n_cells, p['r1'], p['r2'], a=p['a'])
        delta_e = simple_dispersion(k2, p['t0'], a=p['a']) - simple_dispersion(k1, p['t0'], a=p['a'])
        delta_e_case1.append(delta_e)

        psi_i = simple_bloch_state(n_cells, k1, a=p['a'])
        psi_j = simple_bloch_state(n_cells, k2, a=p['a'])
        q_index = (m1 - m2) % n_cells
        q0 = 2.0 * np.pi * q_index / (n_cells * p['a'])
        g_q0 = g_monatomic(psi_i, psi_j, n_cells, q0, p['alpha'], a=p['a'], mass=p['mass'])
        gq0_case1.append(np.abs(g_q0)**2)

    # 上排: 简化版诊断
    axes[0, 0].plot(n_vals, delta_e_case1, 'o-', c=COLORS['case1'])
    axes[0, 0].set_xlabel('$N$')
    axes[0, 0].set_ylabel(r'$\Delta E$')
    axes[0, 0].set_title(r'(a) Energy gap $\Delta E(N)$', fontsize=10)
    axes[0, 0].set_xscale('log')

    axes[0, 1].loglog(n_vals, gq0_case1, 'o-', c=COLORS['case1'])
    # 理论线：当前 g 定义不含 1/sqrt(N)，因此 |g(q0)|^2 ~ 常数
    axes[0, 1].plot(n_vals, np.full_like(n_vals, gq0_case1[0], dtype=float), ':', c=COLORS['theory'], label=r'$\propto N^{0}$')
    # 当数据在机器精度内近似常数时，log-log 的自动缩放可能会把 1e-16 量级的浮点噪声“放大”为可见锯齿；
    # 这里固定一个合理的 y 轴范围，避免误导读者。
    y0 = float(np.mean(gq0_case1))
    if y0 > 0:
        axes[0, 1].set_ylim(y0 * 0.5, y0 * 2.0)
    axes[0, 1].set_xlabel('$N$')
    axes[0, 1].set_ylabel(r'$|g(q_0)|^2$')
    axes[0, 1].set_title('(b) Single-mode coupling', fontsize=10)
    axes[0, 1].legend(fontsize=8)

    # 归一化自检
    norms = []
    for n_cells in n_vals:
        q_vals = np.array([2.0 * np.pi / (n_cells * p['a'])])
        from src.phonon_1atom import displacement_from_q
        q_amp = np.array([1.0 + 0.0j])
        u = displacement_from_q(q_vals, q_amp, n_cells, a=p['a'], mass=p['mass'])
        norms.append(np.sum(np.abs(u)**2))

    axes[0, 2].plot(n_vals, norms, 'o-', c=COLORS['case1'])
    axes[0, 2].axhline(y=norms[0], linestyle=':', c=COLORS['theory'])
    axes[0, 2].set_xlabel('$N$')
    axes[0, 2].set_ylabel(r'$\sum_n |u_n|^2$')
    axes[0, 2].set_title('(c) Normalization check', fontsize=10)
    axes[0, 2].set_xscale('log')

    # 下排: SSH 诊断 - 使用 r_k 定义的 k 点，与 case1_scaling 一致
    ps = SSH_PARAMS
    delta_e_ssh = []
    gq0_ssh = []
    r_k = ps['r_ext']  # 使用与 case1 相同的 k 点定义

    for n_cells in n_vals:
        # 使用远离 BZ 边界的 k 点，与 case1_scaling 一致
        k_ext, _ = select_k_from_ratio(n_cells, r_k, a=ps['a'])
        evals_k, evecs_k = bloch_eigensystem(k_ext, t0=ps['t0'], delta_t=ps['delta_t'], a=ps['a'])
        delta_e_ssh.append(evals_k[1] - evals_k[0])

        psi_v = bloch_state(n_cells, k_ext, evecs_k[:, 0], a=ps['a'])
        psi_c = bloch_state(n_cells, k_ext, evecs_k[:, 1], a=ps['a'])
        omega_q, evec_q = diatomic_modes(0.0, ps['k_spring'], ps['mass_a'], ps['mass_b'], a=ps['a'])
        mode_idx = int(np.argmax(omega_q))
        g_val = g_ssh_diatomic(psi_v, psi_c, n_cells, q=0.0, evec=evec_q[:, mode_idx],
                               alpha=ps['alpha'], a=ps['a'], mass_a=ps['mass_a'], mass_b=ps['mass_b'])
        gq0_ssh.append(np.abs(g_val)**2)

    axes[1, 0].plot(n_vals, delta_e_ssh, 'o-', c=COLORS['case2'])
    axes[1, 0].set_xlabel('$N$')
    axes[1, 0].set_ylabel(r'$\Delta E$')
    axes[1, 0].set_title(r'(d) SSH $\Delta E(N)$ at $k=2\pi r_k/a$', fontsize=10)
    axes[1, 0].set_xscale('log')

    axes[1, 1].loglog(n_vals, gq0_ssh, 'o-', c=COLORS['case2'])
    # 理论线：当前 g 定义不含 1/sqrt(N)，因此 |g(q=0)|^2 ~ 常数
    axes[1, 1].plot(n_vals, np.full_like(n_vals, gq0_ssh[0], dtype=float), ':', c=COLORS['theory'], label=r'$\propto N^{0}$')
    y0 = float(np.mean(gq0_ssh))
    if y0 > 0:
        axes[1, 1].set_ylim(y0 * 0.5, y0 * 2.0)
    axes[1, 1].set_xlabel('$N$')
    axes[1, 1].set_ylabel(r'$|g(q=0)|^2$')
    axes[1, 1].set_title('(e) SSH single-mode coupling', fontsize=10)
    axes[1, 1].legend(fontsize=8)

    # SSH 归一化
    norms_ssh = []
    for n_cells in n_vals:
        q_vals = np.array([0.0])
        omegas, evecs = diatomic_modes_grid(q_vals, ps['k_spring'], ps['mass_a'], ps['mass_b'], a=ps['a'])
        q_amp = np.array([[0.0, 1.0]])
        from src.phonon_diatomic import displacement_from_q as ssh_disp
        disp = ssh_disp(q_vals, q_amp, evecs, n_cells, a=ps['a'], mass_a=ps['mass_a'], mass_b=ps['mass_b'])
        norm = np.sum(np.abs(disp[0])**2) + np.sum(np.abs(disp[1])**2)
        norms_ssh.append(norm)

    axes[1, 2].plot(n_vals, norms_ssh, 'o-', c=COLORS['case2'])
    axes[1, 2].axhline(y=norms_ssh[0], linestyle=':', c=COLORS['theory'])
    axes[1, 2].set_xlabel('$N$')
    axes[1, 2].set_ylabel(r'$\sum |u|^2$')
    axes[1, 2].set_title('(f) SSH normalization check', fontsize=10)
    axes[1, 2].set_xscale('log')

    plt.tight_layout()
    fig.savefig(output_dir / "fig7_diagnostics.png")
    plt.close(fig)
    print("  Figure 7: Diagnostics - done")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """生成所有出版级图表。"""
    print("=" * 60)
    print("NAC 尺寸效应验证 - 生成出版级图表")
    print("=" * 60)

    setup_publication_style()
    output_dir = ensure_output_dir()
    print(f"输出目录: {output_dir}\n")

    print("生成图表...")
    generate_figure1_model_schematic(output_dir)
    generate_figure2_wavefunctions(output_dir)
    simple_results = generate_figure3_scaling_laws(output_dir)
    generate_figure4_gq_distribution(output_dir)
    generate_figure5_folded_phonon(output_dir)
    ssh_results = generate_figure6_ssh_full(output_dir)
    generate_figure7_diagnostics(output_dir)

    print("\n" + "=" * 60)
    print("所有图表生成完成!")
    print("=" * 60)
    print(f"\n简化版结果:")
    print(f"  Case 1 (Ext-Ext): β = {simple_results['case1'][0]:.3f} ± {simple_results['case1'][1]:.3f}")
    print(f"  Case 2 (Loc-Loc): <d²> = {simple_results['case2'][0]:.3e}, σ/μ = {simple_results['case2'][1]:.1f}%")
    print(f"  Case 3 (Loc-Ext): β = {simple_results['case3'][0]:.3f} ± {simple_results['case3'][1]:.3f}")
    print(f"\nSSH版结果:")
    print(f"  Case 1 (Ext-Ext): β = {ssh_results['case1'][0]:.3f} ± {ssh_results['case1'][1]:.3f}")
    print(f"  Case 2 (Loc-Loc): <d²> = {ssh_results['case2'][0]:.3e}, σ/μ = {ssh_results['case2'][1]:.1f}%")
    print(f"  Case 3 (Loc-Ext): β = {ssh_results['case3'][0]:.3f} ± {ssh_results['case3'][1]:.3f}")


if __name__ == "__main__":
    main()
