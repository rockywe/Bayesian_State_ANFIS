import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

# 数据
exp = np.array([
    0.998233436, 0.99788193, 0.99728614, 0.99660573, 0.99338394, 0.970291401,
    0.980531742, 0.983472248, 0.995469955, 0.976606386, 0.981132233, 0.994359495,
    0.996248262, 0.997939695, 0.998502896, 0.999065623, 0.998973397, 0.999254343,
    0.999348179, 0.999442253, 0.999629752, 0.986200206, 0.988007482, 0.992052849,
    0.994450238, 0.996299437, 0.997686419, 0.998334468, 0.998242318, 0.997688223,
    0.997043146, 0.995754057, 0.992895128, 0.99466742, 0.995731216, 0.998219657,
    0.999064619, 0.99750885, 0.996265244, 0.995564881, 0.996454244, 0.976802292,
    0.981895157, 0.986989519, 0.989770304, 0.991624099, 0.99255081, 0.993014156,
    0.993292886, 0.992734699, 0.991622288, 0.989770304, 0.987923139, 0.823455897,
    0.862674582, 0.920538002, 0.962702609, 0.970547694, 0.951002804, 0.930980561
])

sim = np.array([
    0.9868, 0.9847, 0.9834, 0.9818, 0.9784, 0.9904, 0.9909, 0.9912, 0.9919, 0.9859,
    0.9862, 0.9864, 0.9866, 0.9867, 0.9868, 0.9869, 0.9871, 0.9872, 0.9874, 0.9876,
    0.9879, 0.9837, 0.9838, 0.9841, 0.9842, 0.9843, 0.9844, 0.9845, 0.9846, 0.9847,
    0.9848, 0.9849, 0.9853, 0.9858, 0.9863, 0.9867, 0.9868, 0.987, 0.9872, 0.9874,
    0.9875, 0.9866, 0.987, 0.9873, 0.9876, 0.9879, 0.9881, 0.9884, 0.9887, 0.9889,
    0.9891, 0.9893, 0.9895, 0.9875, 0.9878, 0.9889, 0.9893, 0.9897, 0.9899, 0.9905
])

# 计算预测误差百分比
error1 = ((exp - sim) / exp) * 100

# 计算误差的均值和标准差
mean1 = np.mean(error1)
std1 = np.std(error1)

# 拟合一阶多项式（线性拟合）
p = np.polyfit(exp * 100, sim * 100, 1)

# 生成用于绘图的x值
x = np.arange(-30, 30.1, 0.1)

# 拟合分布并计算PDF
y1 = norm(loc=mean1, scale=std1)
y = y1.pdf(x)

# 绘制概率密度函数
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x, y, linewidth=2)
plt.xlabel('Prediction error (%)')
plt.ylabel('Probability density')
plt.title('Probability Density Function of Prediction Error')

# 多项式预测及置信区间
exp_x = sm.add_constant(exp * 100)
model = sm.OLS(sim * 100, exp_x).fit()
st, data, ss2 = summary_table(model, alpha=0.05)

fitted_values = data[:, 2]
predict_ci_low, predict_ci_upp = data[:, 4:6].T

plt.subplot(1, 2, 2)
plt.fill_between(exp * 100, predict_ci_low, predict_ci_upp, color='lightblue', alpha=0.5, label='Prediction Interval')
plt.plot(exp * 100, fitted_values, color='r', label='Fitted Line', linestyle='-')
plt.xlabel('Experimental Data * 100')
plt.ylabel('Simulated Data * 100')
plt.title('Linear Fit with Prediction Interval')
plt.legend()

plt.tight_layout()
plt.show()
