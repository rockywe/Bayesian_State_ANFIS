import numpy as np
import matplotlib.pyplot as plt

# 定义三角隶属函数
def trimf(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# 定义连续变量
pH_cont = np.linspace(11.3, 13.7, 500)
rpm_cont = np.linspace(190, 1700, 500)
ch2o2_cont = np.linspace(0, 1.7, 500)

# 计算隶属度
y111 = trimf(pH_cont, [11.3, 11.8, 12.45])
y222 = trimf(pH_cont, [12.3, 12.9, 13.7])

y333 = trimf(rpm_cont, [190, 495, 760])
y444 = trimf(rpm_cont, [750, 1320, 1700])

y555 = trimf(ch2o2_cont, [0, 0.2, 0.5])
y666 = trimf(ch2o2_cont, [0.4, 0.65, 0.85])
y777 = trimf(ch2o2_cont, [0.8, 1.2, 1.7])

# 绘制隶属度函数图
plt.figure(figsize=(18, 6))

# 吸收剂pH隶属函数
plt.subplot(1, 3, 1)
plt.plot(pH_cont, y111, 'b-', label='Small')
plt.plot(pH_cont, y222, 'r-', label='Big')
plt.xlabel('Absorbent pH', fontsize=23, fontname='Times New Roman')
plt.ylabel('Membership', fontsize=23, fontname='Times New Roman')
plt.title('(a)', fontsize=35)
plt.legend(loc='upper right', frameon=False)

# 转速隶属函数
plt.subplot(1, 3, 2)
plt.plot(rpm_cont, y333, 'b-', label='Small')
plt.plot(rpm_cont, y444, 'r-', label='Big')
plt.xlabel('Rotating speed', fontsize=23, fontname='Times New Roman')
plt.ylabel('Membership', fontsize=23, fontname='Times New Roman')
plt.title('(b)', fontsize=35)
plt.legend(loc='upper right', frameon=False)

# H2O2隶属函数
plt.subplot(1, 3, 3)
plt.plot(ch2o2_cont, y555, 'b-', label='Small')
plt.plot(ch2o2_cont, y666, 'k-', label='Medium')
plt.plot(ch2o2_cont, y777, 'r-', label='Big')
plt.xlabel('${H_2O_2}$ (M)', fontsize=23, fontname='Times New Roman')
plt.ylabel('Membership', fontsize=23, fontname='Times New Roman')
plt.title('(c)', fontsize=35)
plt.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.show()
