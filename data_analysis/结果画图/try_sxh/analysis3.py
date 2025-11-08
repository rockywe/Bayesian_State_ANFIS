import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from mpl_toolkits.mplot3d import Axes3D

# 定义三角隶属函数
def trimf(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# 数据
y_135 = np.array([0.5976, 0.8068, 0.8996, 0.9222, 0.9882, 0.9918, 0.9932, 0.994, 0.993, 0.9922])
y_125 = np.array([0.4642, 0.6564, 0.8052, 0.9198, 0.9074, 0.9116, 0.9148, 0.9106, 0.9078, 0.9056])
y_115 = np.array([0.3332, 0.4426, 0.4976, 0.5312, 0.5666, 0.5922, 0.5788, 0.5832, 0.565, 0.5634])

pH1 = np.array([13.5] * 10 + [12.5] * 10 + [11.5] * 10)
h2o21 = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6] * 3)
y1 = np.concatenate([y_135, y_125, y_115])

# 隶属度计算
y111 = trimf(pH1, [10.0, 11.5, 12.0])
y222 = trimf(pH1, [11.95, 13.5, 14.0])

y333 = trimf(h2o21, [0, 0.2, 0.5])
y444 = trimf(h2o21, [0.4, 0.65, 0.85])
y555 = trimf(h2o21, [0.8, 1.2, 1.7])

# 模糊规则
mem_1 = y111 * y333
mem_2 = y111 * y444
mem_3 = y111 * y555
mem_4 = y222 * y333
mem_5 = y222 * y444
mem_6 = y222 * y555

# 计算β值
mem_sum = mem_1 + mem_2 + mem_3 + mem_4 + mem_5 + mem_6
beta_1 = mem_1 / mem_sum
beta_2 = mem_2 / mem_sum
beta_3 = mem_3 / mem_sum
beta_4 = mem_4 / mem_sum
beta_5 = mem_5 / mem_sum
beta_6 = mem_6 / mem_sum

# 构建大矩阵Y
Y = np.vstack([
    beta_1, beta_2, beta_3, beta_4, beta_5, beta_6,
    beta_1 * pH1, beta_2 * pH1, beta_3 * pH1, beta_4 * pH1, beta_5 * pH1, beta_6 * pH1,
    beta_1 * h2o21, beta_2 * h2o21, beta_3 * h2o21, beta_4 * h2o21, beta_5 * h2o21, beta_6 * h2o21
]).T

# 计算P矩阵
P = pinv(Y.T @ Y) @ Y.T @ y1

# 生成用于绘图的网格
pph = np.linspace(11.5, 13.5, 100)
h22o2 = np.linspace(0.05, 1.6, 100)
z, yyy = np.meshgrid(pph, h22o2)

y11 = trimf(z, [10.0, 11.5, 11.8])
y22 = trimf(z, [11.75, 13.5, 14.0])

y33 = trimf(yyy, [0, 0.2, 0.5])
y44 = trimf(yyy, [0.4, 0.65, 0.85])
y55 = trimf(yyy, [0.8, 1.2, 1.7])

# 计算模糊输出
mem111 = y11 * y33
mem222 = y11 * y44
mem333 = y11 * y55
mem444 = y22 * y33
mem555 = y22 * y44
mem666 = y22 * y55

yy = (mem111 * (P[0] + P[6] * z + P[12] * yyy) +
      mem222 * (P[1] + P[7] * z + P[13] * yyy) +
      mem333 * (P[2] + P[8] * z + P[14] * yyy) +
      mem444 * (P[3] + P[9] * z + P[15] * yyy) +
      mem555 * (P[4] + P[10] * z + P[16] * yyy) +
      mem666 * (P[5] + P[11] * z + P[17] * yyy)) / (mem111 + mem222 + mem333 + mem444 + mem555 + mem666)

# 启用交互模式
plt.ion()

# 绘制3D表面图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(z, yyy, yy, cmap='viridis')
ax.set_xlabel('Absorbent pH', fontsize=14, fontname='Times New Roman')
ax.set_ylabel('${H_2O_2}$ (mol/L)', fontsize=14, fontname='Times New Roman')
ax.set_zlabel('NO removal efficiency (%)', fontsize=14, fontname='Times New Roman')
ax.set_title('Fuzzy Inference Surface', fontsize=16, fontname='Times New Roman')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.show()
