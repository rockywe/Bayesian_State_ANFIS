import numpy as np
from scipy.special import erf

# Constants
r1 = 42e-3  # 内径，单位为米
r2 = 146e-3  # 外径，单位为米
h = 20e-3  # 高度，单位为米
P = 101.325  # 压力，单位为千帕
# H = 709.1735  # Henry常数，单位为 KPa*m3/mol
H = 375.28
packing_area = 500  # 填料比表面积，单位为 m^2/m^3
packing_volume = np.pi * (r2**2 - r1**2) * h  # 填料体积，单位为立方米
r = np.sqrt(r1 * r2)  # 几何平均半径，单位为米
D_no = 2.21e-9  # NO在溶液中的扩散系数，单位为 m^2/s
g_flow = 2 / 3600  # 气体流量，单位为立方米每秒

# Variables
rpm = 1200  # 转速
l_flow = 20e-3 / 3600  # 液体流量
ch2o2 = 0.4  # 氢过氧化物浓度
pH = 12.01  # pH值

# 计算解离程度和氢过氧负离子浓度
al = 10 ** (11.73 - pH)
a = 1 / (1 + al)
ooh = ch2o2 * a

# 表观速率常数
k_noooh = 103.4  # 速率常数
k_apparent = ooh * k_noooh

flow_area = np.pi * (1.6e-3)**2  # 横截面积
l_speed = l_flow / flow_area  # 液相流速

y_in = 0.0005  # 初始浓度
concentration_ratio = (1 - y_in) / y_in  # 浓度比例
packing_flow_factor = packing_area * np.pi * h * (r2**2 - r1**2) / g_flow  # 用来算出口体积

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
print("出口 yout:", yout)

# 去除效率
remove = (y_in - yout) / y_in
print("去除效率 remove:", remove)
