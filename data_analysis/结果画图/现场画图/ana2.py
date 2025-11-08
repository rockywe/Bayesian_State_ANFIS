import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取Excel文件并选择Sheet2工作表
file_path = '魏桥超重力 - 数据处理.xlsx'  
sheet2_data = pd.read_excel(file_path, sheet_name='Sheet2')

# 将时间列转换为日期时间格式（如果需要，假设是时间格式）
# 如果时间是整数序列，则无需进行时间格式转换
# sheet2_data['时间'] = pd.to_datetime(sheet2_data['时间'], format='%H:%M:%S:%f', errors='coerce')

# 创建图形并设置大小
plt.figure(figsize=(10, 6))

# 绘制H2S浓度、进气量和进液量随时间变化的折线图
plt.plot(sheet2_data['时间'], sheet2_data['H2S浓度'], label='H2S浓度', marker='o')
plt.plot(sheet2_data['时间'], sheet2_data['进气量'], label='进气量', marker='s')
plt.plot(sheet2_data['时间'], sheet2_data['进液量'], label='进液量', marker='^')

# 添加标签和标题
plt.xlabel('时间')
plt.ylabel('值')
plt.title('H2S浓度、进气量、进液量随时间变化')
plt.legend()

# 旋转x轴标签以便更好地显示
plt.xticks(rotation=0)
plt.tight_layout()

# 显示图形
plt.show()
