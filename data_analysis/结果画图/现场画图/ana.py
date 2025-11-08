import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取Excel文件并选择Sheet2工作表
file_path = '魏桥超重力 - 数据处理.xlsx'
sheet2_data = pd.read_excel(file_path, sheet_name='Sheet2')

# 将时间列转换为日期时间格式
sheet2_data['时间'] = pd.to_datetime(sheet2_data['时间'], format='%H:%M:%S:%f', errors='coerce')

# 创建图形并设置大小
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制H2S浓度和进液量随时间变化的折线图
ax1.plot(sheet2_data['时间'], sheet2_data['H2S浓度'], label='H2S浓度', color='blue', marker='o')
ax1.plot(sheet2_data['时间'], sheet2_data['进液量'], label='进液量', color='green', marker='^')

# 设置第一个y轴的标签
ax1.set_ylabel('H2S浓度 & 进液量')
ax1.set_xlabel('时间')

# 创建次坐标轴，并绘制进气量的数据
ax2 = ax1.twinx()
ax2.plot(sheet2_data['时间'], sheet2_data['进气量'], label='进气量', color='orange', marker='s')
ax2.set_ylabel('进气量')

# 旋转x轴标签，避免重叠
plt.xticks(rotation=45)

# 设置时间格式显示
ax1.xaxis_date()  # 确保x轴识别时间格式

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 调整布局并显示图形
plt.tight_layout()
plt.show()
