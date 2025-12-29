# 1D Tight-Binding NAC 尺寸效应验证

本项目用于验证非绝热耦合（NAC）的尺寸标度律。

## 核心结论

| Case | 类型 | 理论预期 | 数值结果 |
|------|------|----------|----------|
| 1 | 延展-延展 | β = 1 | β = 1.00, R² = 1.000 |
| 2 | 局域-局域 | β = 0 | σ/μ < 10⁻³ |
| 3 | 局域-延展 | β = 1 | β = 1.00, R² = 1.000 |

解析方法与有限差分数值方法相对误差 < 10⁻⁸。

## 目录结构

```
nac_tb_demo/
├── src/                    # 核心计算模块
│   ├── lattice.py          # 晶格、周期边界、k/q 网格
│   ├── tb_electron_1band.py # 单带 TB 模型
│   ├── ssh_electron.py     # SSH 双带模型
│   ├── phonon_1atom.py     # 单原子链声子
│   ├── phonon_diatomic.py  # 双原子链声子
│   ├── electron_phonon.py  # 电声耦合矩阵元
│   ├── nac.py              # NAC 热平均计算
│   ├── numerical_nac.py    # 有限差分数值验证
│   ├── diagnostics.py      # IPR、态选择、诊断
│   ├── experiments_simple.py    # 单带模型实验
│   ├── experiments_ssh.py       # SSH 模型实验
│   └── experiments_numerical_check.py  # 数值验证实验
├── scripts/                # 可执行脚本
│   ├── generate_publication_figures.py  # 生成出版级图表
│   └── verify_numerical_nac.py          # 数值验证可视化
├── tests/                  # 单元测试
├── results/                # 输出
│   └── publication_figures/  # 出版级图表 (8 张)
└── report/                 # LaTeX 报告
```

## 快速开始

### 依赖

```bash
pip install numpy scipy matplotlib pytest
```

### 运行测试

```bash
pytest -q
```

### 生成图表

```bash
python scripts/generate_publication_figures.py
```

输出 8 张出版级图表到 `results/publication_figures/`：
- `fig1_model_schematic.png` - 模型示意图
- `fig2_wavefunctions.png` - 波函数与 IPR 谱
- `fig3_scaling_laws.png` - 标度律主图（含汇总表）
- `fig4_gq_distribution.png` - 电声耦合分布
- `fig5_folded_phonon.png` - 折叠声子验证
- `fig6_ssh_scaling.png` - SSH 模型标度律
- `fig7_diagnostics.png` - 诊断图
- `fig_appendix_numerical_verification.png` - 数值验证

### 数值验证

```bash
python scripts/verify_numerical_nac.py
```

## 物理模型

### 单带紧束缚模型

哈密顿量：
$$H = -t_0 \sum_n (c_n^\dagger c_{n+1} + \text{h.c.})$$

色散关系：$E(k) = -2t_0 \cos(ka)$

### SSH 双带模型

交替跳跃积分 $v = t_0 + \delta t$（胞内），$w = t_0 - \delta t$（胞间）。

能带：$E_\pm(k) = \pm\sqrt{v^2 + w^2 + 2vw\cos(ka)}$

### NAC 标度律

$$\langle d^2 \rangle = \frac{1}{(\Delta E)^2} \sum_q |g(q)|^2 \langle \dot{Q}_q^2 \rangle$$

- **Case 1 (Ext-Ext)**：动量守恒 → δ 峰 → $\langle d^2 \rangle \propto N^{-1}$
- **Case 2 (Loc-Loc)**：局域扰动 → $\langle d^2 \rangle \propto N^0$
- **Case 3 (Loc-Ext)**：混合 → $\langle d^2 \rangle \propto N^{-1}$

## 参数设置

| 参数 | 值 |
|------|-----|
| 系统尺寸 N | 20, 40, 60, 80, 120, 160, 240, 320, 480, 640 |
| 跳跃积分 t₀ | 1.0 |
| SSH 交替参数 δt | 0.2 |
| 电声耦合 α | 0.5 |
| 温度 T | 300 K |
| 缺陷宽度 | 5 格点 |
| 缺陷深度 | -1.5 t₀ |
| IPR 阈值 | 0.05 |
