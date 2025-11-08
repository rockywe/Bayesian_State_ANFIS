import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sanfis import SANFIS, plottingtools # 导入 plottingtools
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

is_available = torch.cuda.is_available()
print(f"CUDA available: {is_available}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# # 加载数据
# file_path = './data/脱硫数据整理2.xlsx'
# try:
#     data_df = pd.read_excel(file_path, sheet_name='Sheet1')
# except FileNotFoundError:
#     print(f"错误：文件未找到，请检查路径：{file_path}")
#     exit() # 退出程序或进行其他错误处理
#
# # 创建中英文列名映射字典
# column_rename_dict = {
#     '煤气进口流量': 'Gas_Inlet_Flow',
#     '进口煤气温度': 'Gas_Inlet_Temperature',
#     '进口煤气压力': 'Gas_Inlet_Pressure',
#     '脱硫液流量': 'Desulfurization_Liquid_Flow',
#     '脱硫液温度': 'Desulfurization_Liquid_Temperature',
#     '脱硫液压力': 'Desulfurization_Liquid_Pressure',
#     '转速': 'Rotation_Speed',
#     '进口H2S浓度': 'H2S_Inlet_Concentration',
#     '出口H2S浓度': 'H2S_Outlet_Concentration'
# }
#
# # 替换列名
# data_df.rename(columns=column_rename_dict, inplace=True)
#
# # 设置英文版输入输出特征名
# input_features = [
#     'Gas_Inlet_Flow', 'Gas_Inlet_Temperature', 'Gas_Inlet_Pressure',
#     'Desulfurization_Liquid_Flow', 'Desulfurization_Liquid_Temperature',
#     'Desulfurization_Liquid_Pressure', 'Rotation_Speed',
#     'H2S_Inlet_Concentration'
# ]
# output_feature = 'H2S_Outlet_Concentration'
#
# # 检查数据是否为空
# if data_df.empty:
#     print("错误：加载的数据为空。请检查Excel文件内容。")
#     exit()
#
# X = data_df[input_features].values
# y = data_df[output_feature].values.reshape(-1, 1)
#
# # 数据标准化
# scaler_X = MinMaxScaler()
# scaler_y = MinMaxScaler()
# X_scaled = scaler_X.fit_transform(X)
# y_scaled = scaler_y.fit_transform(y)
#
# # 划分训练集和测试集
# X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
#     X_scaled, y_scaled, test_size=0.2, random_state=42)
#
# # 将 NumPy 数组转换为 PyTorch 张量
# X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)
#
# # SANFIS 需要 'S' (状态) 输入。如果没有单独的状态数据，通常将输入特征 X 作为 S。
# S_train_tensor = X_train_tensor
# S_test_tensor = X_test_tensor
#
# # 为 SANFIS 的 fit 方法准备数据列表: [S_data, X_data, y_data]
# train_data = [S_train_tensor, X_train_tensor, y_train_tensor]
# # SANFIS 的 fit 方法接受验证集，这里使用测试集作为验证集
# valid_data = [S_test_tensor, X_test_tensor, y_test_tensor]
#
# # --- SANFIS 模型参数配置 ---
# n_input_features = X_scaled.shape[1] # 输入特征的数量
# n_memb_funcs_per_input = 3 # 每个输入特征的隶属函数数量
#
# # 构建 membfuncs 列表（用于定义隶属函数）
# membfuncs_config = []
# # 为高斯隶属函数的 mu (均值) 生成初始值，均匀分布在 [0, 1] 范围内
# initial_mu_values = np.linspace(0, 1, n_memb_funcs_per_input).tolist()
# # 为高斯隶属函数的 sigma (标准差) 设置一个小的初始值
# initial_sigma_value = 0.1
#
# for _ in range(n_input_features):
#     membfuncs_config.append({
#         'function': 'gaussian',
#         'n_memb': n_memb_funcs_per_input,
#         'params': {
#             'mu': {'value': initial_mu_values, 'trainable': True},
#             'sigma': {'value': [initial_sigma_value] * n_memb_funcs_per_input, 'trainable': True}
#         }
#     })
#
# # 初始化 SANFIS 模型
# model = SANFIS(
#     membfuncs=membfuncs_config,
#     n_input=n_input_features,
#     to_device='cpu', # 明确指定在CPU上运行，避免GPU配置问题
#     scale='Std' # 根据推荐代码，使用 'Std' 缩放
# )
#
# # 定义损失函数和优化器
# loss_function = torch.nn.MSELoss(reduction='mean')
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # 使用推荐的学习率
#
# print(f"\n开始训练 SANFIS 模型...")
#
# # 训练模型并获取 history 对象
# history = model.fit(
#     train_data,
#     valid_data, # 传递验证数据
#     optimizer,
#     loss_function,
#     epochs=100 # 训练轮次
# )
#
# print(f"SANFIS 模型训练完成。")
#
# # 在测试集上进行预测
# # SANFIS 的 predict 方法也期望 [S, X]
# y_pred_scaled_tensor = model.predict([S_test_tensor, X_test_tensor])
#
# # 将预测结果转换回 NumPy 数组，以便进行反标准化和评估
# y_pred_scaled = y_pred_scaled_tensor.detach().numpy()
#
# # 反标准化预测结果和真实值
# y_pred = scaler_y.inverse_transform(y_pred_scaled)
# y_test_original = scaler_y.inverse_transform(y_test_np) # 注意这里用回了y_test_np
#
# # 计算误差
# mse = mean_squared_error(y_test_original, y_pred)
# r2 = r2_score(y_test_original, y_pred)
#
# print(f"\n模型评估结果 (基于测试集):")
# print(f"均方误差 (MSE): {mse:.4f}")
# print(f"决定系数 (R²): {r2:.4f}")


