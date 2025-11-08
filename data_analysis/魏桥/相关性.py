import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 步骤 1: 读取CSV文件中的数据
data = pd.read_excel('魏桥超重力.xlsx')

# 步骤 2: 选择你感兴趣的相关参数（可以根据实际情况调整）
parameters = ['进口煤气流量', '进口煤气温度', '进口煤气压力',
              '脱硫液进口调节阀开度', '脱硫液流量', '脱硫液进口温度',
              '脱硫液进口压力', '电流', '频率',
              '液泵频率', '液泵电流', '出口煤气温度', 'H2S浓度']

# 步骤 3: 计算皮尔逊相关系数矩阵
corr_matrix = data[parameters].corr(method='pearson')

# 步骤 4: 可视化相关性矩阵 - 使用热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('参数相关性热力图')
plt.show()

# 步骤 5: 可视化一些关键参数的散点图 - 例如 进口煤气流量与H2S浓度
sns.scatterplot(x='进口煤气流量', y='H2S浓度', data=data)
plt.title('进口煤气流量与H₂S浓度的散点图')
plt.xlabel('进口煤气流量 (方)')
plt.ylabel('H₂S浓度 (mg/m³)')
plt.show()

# 你也可以选择其它参数进行散点图可视化
sns.scatterplot(x='脱硫液流量', y='H2S浓度', data=data)
plt.title('脱硫液流量与H₂S浓度的散点图')
plt.xlabel('脱硫液流量 (L/min)')
plt.ylabel('H₂S浓度 (mg/m³)')
plt.show()
