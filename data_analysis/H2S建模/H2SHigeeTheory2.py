import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import math
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, AutoLocator

# 定义函数
def k_ov(w):
    """计算旋转速度w（RPM）下的k_ov值"""
    return 1 * (0.0041 * np.exp(w * np.pi * R / 30) + 0.21746)

def N_chem(c_H2S, k_ov_val, D_H2S, bar_t_val):
    """计算化学通量N_H2S"""
    term1 = bar_t_val * erf(np.sqrt(k_ov_val * bar_t_val))
    term2 = np.sqrt(bar_t_val / (np.pi * k_ov_val)) * np.exp(-k_ov_val * bar_t_val)
    term3 = (1 / (2 * k_ov_val)) * erf(np.sqrt(k_ov_val * bar_t_val))
    return c_H2S * np.sqrt(k_ov_val * D_H2S) / bar_t_val * (term1 + term2 + term3)

def k_L(N_chem_val, c_H2S_star, c_H2S_0):
    """计算传质系数k_L"""
    return N_chem_val / (c_H2S_star - c_H2S_0)

def bar_t(R_out, R_in, L, w, R, N_S):
    """计算平均停留时间bar_t"""
    return (R_out - R_in) / (0.0217 * L**0.2279 * (w * R)**0.5448 * N_S)

def y_sim(y_in, k_L_val, T, P, H_H2S, alpha, pi, h, R_out, R_in, G_N2):
    """计算模拟出口浓度y_sim"""
    exponent = k_L_val * 0.082 * T * (P / H_H2S) * alpha * pi * h * (R_out**2 - R_in**2) / G_N2
    print(f"Exponent: {exponent}")  # 调试输出
    return y_in / ((1 - y_in) * np.exp(exponent) + y_in)

# 常数和初始参数

D_H2S = 2.5e-9        # H2S的扩散系数 (m²/s)
R_in = 0.015           # 内半径 (m)
R_out = 0.85           # 外半径 (m)

c_H2S_star = 500e-6    # H2S的平衡浓度 (ppm)
c_H2S_0 = 0            # 液相中初始H2S浓度 (ppm)
N_S = 100              # 网格数
R = np.sqrt(R_out * R_in)  # 平均半径 (m)
h = 2                  # 高度 (m)
P = (900e-6) * 101.325 # 分压 (kPa)
H_H2S = 725            # 亨利常数 (KPa·m³/mol)
alpha = 500            # 比表面积 (m²/m³)

# 变量
y_in = 900 * 1e-6     # ppm
G_N2 = 0.3         # 气体流量 (m³/s)1200/3600

# 设置全局字体和字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 全局字体大小

# 情况一：改变转速 w 从 600 到 1200 RPM
w_values = np.linspace(600, 1200, 100)
T_fixed = 298.15     # 固定温度 (K)
GLR_fixed = 40      # 固定气液比
L_fixed = G_N2 / GLR_fixed  # 对应的液体流量 (m³/s)
y_sim_w = []

for w in w_values:
    k_ov_val = k_ov(w + 400)  # 调整转速加400
    bar_t_val = bar_t(R_out, R_in, L_fixed, w + 300, R, N_S)  # 调整转速加300
    N_chem_val = N_chem(y_in, k_ov_val, D_H2S, bar_t_val)
    k_L_val = k_L(N_chem_val, c_H2S_star, c_H2S_0)
    y_sim_val = 1e6 * y_sim(y_in, k_L_val, T_fixed, P, H_H2S, alpha, np.pi, h, R_out, R_in, G_N2)
    y_sim_w.append(y_sim_val)

    # 调试输出
    print(f"w: {w}, k_ov_val: {k_ov_val}, bar_t_val: {bar_t_val}, N_chem_val: {N_chem_val}, k_L_val: {k_L_val}, y_sim_val: {y_sim_val}")

# 绘图
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(w_values, y_sim_w, color='b', linewidth=2)
plt.xlabel('Rotating Speed (RPM)', fontsize=14)
plt.ylabel('$y_{\mathrm{sim}}$ (ppm)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置刻度朝外
plt.tick_params(axis='both', which='major', direction='out', length=6, width=1)
plt.tick_params(axis='both', which='minor', direction='out', length=4, width=1)

# 设置主、副刻度
plt.gca().xaxis.set_major_locator(MultipleLocator(200))  # 主刻度间隔
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())  # 副刻度
plt.gca().yaxis.set_major_locator(AutoLocator())  # 主刻度间隔
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())  # 副刻度

plt.tight_layout()
plt.savefig('y_sim_w.pdf', format='pdf', bbox_inches='tight')
plt.savefig('y_sim_w.png', dpi=2400, bbox_inches='tight')
plt.show()
