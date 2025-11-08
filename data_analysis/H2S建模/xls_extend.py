import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# 读取原始Excel文件
file_path = 'H2S脱除.xlsx'
df = pd.read_excel(file_path)

# 显示原始数据
print("原始数据:")
print(df.head())

# 提取原始数据列
rpm = df['rpm']
l_flow = df['l_flow']
ph = df['ph']
tannic_conc = df['tannic_conc']
sodium_conc = df['sodium_conc']
efficiency = df['y']

# 组合输入变量
input_variables = np.array([rpm, l_flow, ph, tannic_conc, sodium_conc]).T

# 创建多维插值函数
interpolator = LinearNDInterpolator(input_variables, efficiency)

# 生成新的插值数据点
new_rpm = np.linspace(rpm.min(), rpm.max(), num=10)
new_l_flow = np.linspace(l_flow.min(), l_flow.max(), num=10)
new_ph = np.linspace(ph.min(), ph.max(), num=10)
new_tannic_conc = np.linspace(tannic_conc.min(), tannic_conc.max(), num=10)
new_sodium_conc = np.linspace(sodium_conc.min(), sodium_conc.max(), num=10)

# 创建网格
new_grid = np.array(np.meshgrid(new_rpm, new_l_flow, new_ph, new_tannic_conc, new_sodium_conc)).T.reshape(-1, 5)

# 使用插值函数计算新的去除效率数据点
new_efficiency = interpolator(new_grid)

# 过滤掉插值无效的点
valid_points = ~np.isnan(new_efficiency)
new_grid = new_grid[valid_points]
new_efficiency = new_efficiency[valid_points]

# 创建新的DataFrame
new_df = pd.DataFrame(new_grid, columns=['rpm', 'l_flow', 'ph', 'tannic_conc', 'sodium_conc'])
new_df['y'] = new_efficiency

# 显示扩充后的数据
print("扩充后的数据:")
print(new_df.head())

# 保存扩充后的数据到新的Excel文件
new_file_path = 'H2S_latest.xlsx'
new_df.to_excel(new_file_path, index=False)

print(f"扩充后的数据已保存到: {new_file_path}")
