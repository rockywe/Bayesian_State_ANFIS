import numpy as np
from scipy.special import erf

# Constants
#内外径、高、压强、亨利常数
r1 = 42e-3
r2 = 146e-3
h = 20e-3
P = 101.325
H = 709.1735

# 填料的比面积
packing_area = 500
# 填料体积
packing_volume = np.pi * (r2**2 - r1**2) * (20e-3)
# 平均半径
r = np.sqrt(r1 * r2)
# 扩散系数
D_no = 2.21e-9
# 气体流量
g_flow = 2/3600

# 变量
# 转速
rpm = 1200
# 液体流量
l_flow = 20e-3 / 3600
# 超氧浓度
c_oo2 = 7e-9


# 横截面积
flow_area = np.pi * (1.6e-3)**2
# 流速
l_speed = l_flow/flow_area
# 速率常数
k_noooh = 6.9e9
# 表观速率常数
k_apparent = c_oo2 * k_noooh


# 初始浓度
y_in = 0.0005
# 浓度比例
concentration_ratio = (1 - y_in) / y_in
# 用来算出口体积
packing_flow_factor = packing_area * np.pi * h * (r2**2 - r1**2) / g_flow

# 液膜更新时间
def cal_t_new(l_speed, rpm):
    t_new = (r2 - r1) / (0.02107 * (l_speed ** 0.2279) * ((r * rpm)**0.5448) * 31)
    return t_new

# 液膜传质系数
def calculate_transfer_l(D_no, k_apparent, t_new):
    transfer_l = (np.sqrt(D_no * k_apparent) / t_new) * (
        t_new * erf(np.sqrt(k_apparent * t_new)) +
        np.sqrt((t_new / np.pi) / k_apparent) * np.exp(-k_apparent * t_new) +
        (0.5 / k_apparent) * erf(np.sqrt(k_apparent * t_new))
    )
    return transfer_l

t_new = cal_t_new(l_speed, rpm)
transfer_l = calculate_transfer_l(D_no, k_apparent, t_new)


# 气液传质系数
transfer_y = 0.082 * 298.15 * 0.2700 * transfer_l

# 出口
yout = 1 / (concentration_ratio * np.exp(packing_flow_factor * transfer_y) + 1)
print(yout)
remove = (y_in - yout) / y_in
print(remove)


