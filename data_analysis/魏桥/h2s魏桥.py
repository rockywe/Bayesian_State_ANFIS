import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
# 设置中文字体路径，这里以Microsoft YaHei为例，您需要替换为本机有的中文字体文件路径
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=16)
# Load the Excel file again
file_path = '魏桥超重力 - 数据处理2.xlsx'
sheet2_data = pd.read_excel(file_path, sheet_name='Sheet2')

# Extract features and target
X = sheet2_data[['进口煤气流量', '进口液流量', '频率']]
y = sheet2_data['H2S浓度mg/m3']

# Handle missing values
X_filled = X.fillna(X.mean())
y_filled = y.fillna(y.mean())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filled, y_filled, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the neural network
nn_model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)

# Generate the data for the three requested plots
liquid_flow_range = np.linspace(X_filled['进口液流量'].min(), X_filled['进口液流量'].max(), 50)
frequency_range = np.linspace(X_filled['频率'].min(), X_filled['频率'].max(), 50)
gas_flow_range = np.linspace(X_filled['进口煤气流量'].min(), X_filled['进口煤气流量'].max(), 50)

# Create the grid for Plot 1 (Liquid Flow vs Frequency)
liquid_flow_grid, frequency_grid = np.meshgrid(liquid_flow_range, frequency_range)
fixed_gas_flow = X_filled['进口煤气流量'].mean()  # Fixing gas flow to its mean

# Prepare data for Plot 1
grid_data_1 = np.c_[np.full_like(liquid_flow_grid.ravel(), fixed_gas_flow), liquid_flow_grid.ravel(), frequency_grid.ravel()]
grid_data_scaled_1 = scaler.transform(grid_data_1)

# Predict H₂S concentration for Plot 1
predicted_h2s_1 = nn_model.predict(grid_data_scaled_1).reshape(liquid_flow_grid.shape)

# Create the grid for Plot 2 (Liquid Flow vs Liquid Flow)
gas_flow, liquid_flow_grid_y = np.meshgrid(gas_flow_range, liquid_flow_range)
fixed_frequency = X_filled['频率'].mean()  # Fixing frequency to its mean

# Prepare data for Plot 2
grid_data_2 = np.c_[np.full_like(gas_flow.ravel(), fixed_gas_flow), gas_flow.ravel(), np.full_like(liquid_flow_grid_y.ravel(), fixed_frequency)]
grid_data_scaled_2 = scaler.transform(grid_data_2)

# Predict H₂S concentration for Plot 2
predicted_h2s_2 = nn_model.predict(grid_data_scaled_2).reshape(gas_flow.shape)

# Create the grid for Plot 3 (Gas Flow vs Frequency)
gas_flow_grid, frequency_grid_3 = np.meshgrid(gas_flow_range, frequency_range)
fixed_liquid_flow = X_filled['进口液流量'].mean()  # Fixing liquid flow to its mean

# Prepare data for Plot 3
grid_data_3 = np.c_[gas_flow_grid.ravel(), np.full_like(gas_flow_grid.ravel(), fixed_liquid_flow), frequency_grid_3.ravel()]
grid_data_scaled_3 = scaler.transform(grid_data_3)

# Predict H₂S concentration for Plot 3
predicted_h2s_3 = nn_model.predict(grid_data_scaled_3).reshape(gas_flow_grid.shape)

# Plot 1 (Liquid Flow vs Frequency)
fig1 = plt.figure(figsize=(14, 10))
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(liquid_flow_grid, frequency_grid, predicted_h2s_1, cmap='plasma', edgecolor='none')
ax1.set_title('3D可视化(进液量 vs 频率)', fontproperties=font)
ax1.set_xlabel('进液量', fontproperties=font)
ax1.set_ylabel('频率', fontproperties=font)
ax1.set_zlabel('H₂S浓度(mg/m³)', fontproperties=font)
fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

plt.show()

# Plot 2 (Liquid Flow vs Liquid Flow)
fig2 = plt.figure(figsize=(14, 10))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(gas_flow_grid, liquid_flow_grid_y, predicted_h2s_2, cmap='plasma', edgecolor='none')
ax2.set_title('3D可视化(进液量 vs 进气量)', fontproperties=font)
ax2.set_xlabel('进气量', fontproperties=font)
ax2.set_ylabel('进液量', fontproperties=font)
ax2.set_zlabel('H₂S浓度(mg/m³)', fontproperties=font)
fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
plt.show()

# Plot 3 (Gas Flow vs Frequency)
fig3 = plt.figure(figsize=(14, 10))
ax3 = fig3.add_subplot(111, projection='3d')
surf3 = ax3.plot_surface(gas_flow_grid, frequency_grid_3, predicted_h2s_3, cmap='plasma', edgecolor='none')
ax3.set_title('3D可视化(进气量 vs 频率)', fontproperties=font)
ax3.set_xlabel('进气量', fontproperties=font)
ax3.set_ylabel('频率', fontproperties=font)
ax3.set_zlabel('H₂S浓度(mg/m³)', fontproperties=font)
fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
plt.show()

# 等待用户输入，防止程序立即退出
input("Press Enter to exit...")