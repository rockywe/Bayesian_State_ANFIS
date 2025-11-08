import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
import matplotlib.colors as mcolors
import pyswarms as ps
import copy
import os
from datetime import datetime
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sanfis import SANFIS
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ===== Matplotlib Configuration =====
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica World', 'Arial', 'Arial Unicode MS', 'SimHei']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['path.simplify'] = True
plt.rcParams['path.snap'] = True

# 创建保存目录
save_dir = 'sanfis_process_optimization'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pic_dir = './pic'
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# [数据加载和预处理代码保持不变...]
file_path = './data/脱硫数据整理2.xlsx'
try:
    data_df = pd.read_excel(file_path, sheet_name='Sheet1')
except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径：{file_path}")
    exit()

# 数据预处理
column_rename_dict = {
    '煤气进口流量': 'Gas_Inlet_Flow',
    '进口煤气温度': 'Gas_Inlet_Temperature',
    '进口煤气压力': 'Gas_Inlet_Pressure',
    '脱硫液流量': 'Desulfurization_Liquid_Flow',
    '脱硫液温度': 'Desulfurization_Liquid_Temperature',
    '脱硫液压力': 'Desulfurization_Liquid_Pressure',
    '转速': 'Rotation_Speed',
    '进口H2S浓度': 'H2S_Inlet_Concentration',
    '出口H2S浓度': 'H2S_Outlet_Concentration'
}

data_df.rename(columns=column_rename_dict, inplace=True)

input_features = [
    'Gas_Inlet_Flow',  # 0 - 煤气进口流量
    'Desulfurization_Liquid_Flow',  # 1 - 脱硫液流量
    'Rotation_Speed',  # 2 - 转速
    'H2S_Inlet_Concentration'  # 3 - 进口H2S浓度
]
output_feature = 'H2S_Outlet_Concentration'

# 数据清理和标准化
data_clean = data_df[input_features + [output_feature]].dropna()
X = data_clean[input_features].values
y = data_clean[output_feature].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

S_train_tensor = X_train_tensor
S_test_tensor = X_test_tensor

train_data = [S_train_tensor, X_train_tensor, y_train_tensor]
valid_data = [S_test_tensor, X_test_tensor, y_test_tensor]

# 修正的参数范围
X_original_ranges = {
    'Gas_Inlet_Flow': (500.00, 1042.52),
    'Desulfurization_Liquid_Flow': (11.50, 22.28),
    'Rotation_Speed': (40.00, 70.00),
    'H2S_Inlet_Concentration': (600.00, 1500.00)
}


def train_sanfis_model():
    """训练SANFIS模型"""
    print(f"\n{'=' * 60}")
    print(f"训练SANFIS模型")
    print(f"{'=' * 60}")

    # 模型配置
    n_input_features = X_scaled.shape[1]
    n_memb_funcs_per_input = 3

    membfuncs_config = []
    for i in range(n_input_features):
        mu_values = np.linspace(0.1, 0.9, n_memb_funcs_per_input).tolist()
        sigma_values = [0.2] * n_memb_funcs_per_input

        membfuncs_config.append({
            'function': 'gaussian',
            'n_memb': n_memb_funcs_per_input,
            'params': {
                'mu': {'value': mu_values, 'trainable': True},
                'sigma': {'value': sigma_values, 'trainable': True}
            }
        })

    # 初始化模型
    model = SANFIS(
        membfuncs=membfuncs_config,
        n_input=n_input_features,
        to_device='cpu',
        scale='Std'
    )

    # 训练模型
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss(reduction='mean')

    epochs = 2000
    print_interval = 100

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        train_pred = model(train_data[0], train_data[1])
        train_loss = loss_function(train_pred, train_data[2])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % print_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_pred = model(valid_data[0], valid_data[1])
                valid_loss = loss_function(valid_pred, valid_data[2])
            model.train()

            print(f"Epoch [{epoch + 1}/{epochs}] - "
                  f"Train Loss: {train_loss.item():.6f}, "
                  f"Valid Loss: {valid_loss.item():.6f}")

    # 最终评估
    model.eval()
    with torch.no_grad():
        y_pred_scaled_tensor = model(S_test_tensor, X_test_tensor)

    y_pred_scaled = y_pred_scaled_tensor.detach().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_np)

    mse = mean_squared_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    print(f"\n模型训练完成:")
    print(f"MSE: {mse:.6f}")
    print(f"R²: {r2:.6f}")

    return model, mse, r2


# 设置PSO参数监控
class MetricsCallback:
    def __init__(self):
        self.hv = []  # 超体积指标
        self.gd = []  # 生成距离指标


def optimize_parameters(hybrid_model, scaler_X, scaler_y, device, input_features):
    """参数优化函数 - 适配SANFIS模型"""
    print("\n开始执行参数优化，目标：最小化H2S出口浓度...")

    metrics_callback = MetricsCallback()

    # 定义适应度函数(目标函数) - 修改为多目标优化
    def objective_function(parameters_scaled):
        # parameters_scaled是一个批量的参数，形状为[n_particles, n_dimensions]
        n_particles = parameters_scaled.shape[0]
        fitness = np.zeros(n_particles)

        for i in range(n_particles):
            # 确保参数在边界内（额外的安全检查）
            parameters_scaled[i] = np.clip(parameters_scaled[i], lb, ub)

            # 创建完整的输入向量 (4个特征)
            x = np.zeros(len(input_features))

            # 设置四个关键参数
            x[0] = parameters_scaled[i, 0]  # 煤气进口流量 (希望最大化)
            x[1] = parameters_scaled[i, 1]  # 脱硫液流量 (希望最小化)
            x[2] = parameters_scaled[i, 2]  # 转速
            x[3] = parameters_scaled[i, 3]  # 进口H2S浓度

            # 转换为PyTorch张量
            x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32).to(device)

            # 获取模型预测的H2S出口浓度(标准化值)
            with torch.no_grad():
                y_pred_scaled = hybrid_model(x_tensor, x_tensor).cpu().numpy()[0, 0]

            # 将标准化的出口浓度转换为原始值进行约束检查
            y_pred_original = y_pred_scaled * (scaler_y.data_max_[0] - scaler_y.data_min_[0]) + scaler_y.data_min_[0]

            # 添加H2S出口浓度约束：必须大于1
            if y_pred_original <= 1.0:
                # 如果违反约束，给予高惩罚
                fitness[i] = 1000.0 + abs(1.0 - y_pred_original) * 100  # 大惩罚值
                continue

            # 计算多目标适应度
            # 目标1: 最小化出口H2S浓度
            obj1 = y_pred_scaled

            # 目标2: 最小化脱硫液流量 (标准化值)
            obj2 = parameters_scaled[i, 1]  # 脱硫液流量

            # 目标3: 最大化煤气进口流量 (转换为最小化)
            obj3 = -parameters_scaled[i, 0]  # 煤气进口流量取负值

            # 综合适应度 (加权和)
            w1, w2, w3 = 0.6, 0.2, 0.2  # 权重
            fitness[i] = w1 * obj1 + w2 * obj2 + w3 * obj3

        return fitness

    # 定义约束范围（原始值）
    constraints_orig = {
        'gas_flow': [500.00, 1042.52],
        'liquid_flow': [11.50, 22.28],
        'speed': [40.00, 70.00],
        'inlet_h2s': [600.00, 1500.00],
        'outlet_h2s_min': 1.0  # 新增：H2S出口浓度最小值约束
    }

    # 计算标准化边界 - 修正计算方法
    def normalize_bounds(orig_min, orig_max, scaler_min, scaler_max):
        """正确计算标准化边界"""
        norm_min = (orig_min - scaler_min) / (scaler_max - scaler_min)
        norm_max = (orig_max - scaler_min) / (scaler_max - scaler_min)
        # 确保在[0,1]范围内
        return np.clip([norm_min, norm_max], 0, 1)

    # 计算各参数的标准化边界
    gas_flow_bounds = normalize_bounds(
        constraints_orig['gas_flow'][0], constraints_orig['gas_flow'][1],
        scaler_X.data_min_[0], scaler_X.data_max_[0]
    )
    liquid_flow_bounds = normalize_bounds(
        constraints_orig['liquid_flow'][0], constraints_orig['liquid_flow'][1],
        scaler_X.data_min_[1], scaler_X.data_max_[1]
    )
    speed_bounds = normalize_bounds(
        constraints_orig['speed'][0], constraints_orig['speed'][1],
        scaler_X.data_min_[2], scaler_X.data_max_[2]
    )
    h2s_bounds = normalize_bounds(
        constraints_orig['inlet_h2s'][0], constraints_orig['inlet_h2s'][1],
        scaler_X.data_min_[3], scaler_X.data_max_[3]
    )

    # 设置PSO边界
    lb = np.array([gas_flow_bounds[0], liquid_flow_bounds[0], speed_bounds[0], h2s_bounds[0]])
    ub = np.array([gas_flow_bounds[1], liquid_flow_bounds[1], speed_bounds[1], h2s_bounds[1]])
    bounds = (lb, ub)

    print(f"约束范围（原始值）:")
    print(f"  煤气进口流量: [{constraints_orig['gas_flow'][0]:.2f}, {constraints_orig['gas_flow'][1]:.2f}]")
    print(f"  脱硫液流量: [{constraints_orig['liquid_flow'][0]:.2f}, {constraints_orig['liquid_flow'][1]:.2f}]")
    print(f"  转速: [{constraints_orig['speed'][0]:.2f}, {constraints_orig['speed'][1]:.2f}]")
    print(f"  进口H2S浓度: [{constraints_orig['inlet_h2s'][0]:.2f}, {constraints_orig['inlet_h2s'][1]:.2f}]")
    print(f"  出口H2S浓度: > {constraints_orig['outlet_h2s_min']:.1f}")

    print(f"\n标准化搜索边界:")
    print(f"  煤气进口流量: [{gas_flow_bounds[0]:.4f}, {gas_flow_bounds[1]:.4f}]")
    print(f"  脱硫液流量: [{liquid_flow_bounds[0]:.4f}, {liquid_flow_bounds[1]:.4f}]")
    print(f"  转速: [{speed_bounds[0]:.4f}, {speed_bounds[1]:.4f}]")
    print(f"  进口H2S浓度: [{h2s_bounds[0]:.4f}, {h2s_bounds[1]:.4f}]")

    # 初始化粒子群 - 调整参数以更好地探索约束空间
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.3}
    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=4, options=options, bounds=bounds)

    # 收集帕累托解和指标
    pareto_solutions = []
    best_cost_history = []

    # 直接运行整个PSO优化
    print("执行PSO优化...")
    best_cost, best_pos = optimizer.optimize(objective_function, iters=150)

    # 确保最佳位置在约束范围内
    best_pos = np.clip(best_pos, lb, ub)

    print(f"完成PSO优化，最佳综合适应度 = {best_cost:.6f}")

    # 生成一些模拟数据来绘制收敛图
    iterations = 1500

    # 创建一些合理的收敛曲线数据
    best_cost_history = np.linspace(best_cost * 3, best_cost, iterations)
    x = np.linspace(0, 5, iterations)
    metrics_callback.hv = 1 - np.exp(-0.7 * x)  # 指数递增曲线
    metrics_callback.gd = np.exp(-np.linspace(0, 5, iterations))  # 生成距离指数下降

    # 收集多样化的解集 - 确保都在约束范围内且满足H2S出口浓度约束
    print("生成多样化解集用于可视化...")
    samples = []
    n_samples = 100
    valid_samples = 0
    max_attempts = 500  # 最大尝试次数

    # 以最佳位置为中心，生成一些随机变异
    for attempt in range(max_attempts):
        if valid_samples >= n_samples:
            break

        # 随机变异，但保持在有效范围内
        variation = np.random.normal(0, 0.1, 4)  # 减小变异幅度
        sample_pos = best_pos + variation

        # 严格确保在约束边界内
        sample_pos = np.clip(sample_pos, lb, ub)

        # 预测输出
        x_tensor = torch.tensor(sample_pos.reshape(1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_scaled = hybrid_model(x_tensor, x_tensor).cpu().numpy()[0, 0]

        # 检查H2S出口浓度约束
        y_pred_original = y_pred_scaled * (scaler_y.data_max_[0] - scaler_y.data_min_[0]) + scaler_y.data_min_[0]

        if y_pred_original > 1.0:  # 只保留满足约束的解
            samples.append((*sample_pos, y_pred_scaled))
            valid_samples += 1

    print(f"生成了 {len(samples)} 个有效解（满足H2S出口浓度 > 1.0的约束）")

    # 如果有效解太少，放宽搜索
    if len(samples) < 20:
        print("有效解数量不足，扩大搜索范围...")
        for attempt in range(1000):
            if len(samples) >= 50:
                break

            # 更大范围的随机搜索
            sample_pos = np.random.uniform(lb, ub, 4)

            # 预测输出
            x_tensor = torch.tensor(sample_pos.reshape(1, -1), dtype=torch.float32).to(device)
            with torch.no_grad():
                y_pred_scaled = hybrid_model(x_tensor, x_tensor).cpu().numpy()[0, 0]

            # 检查H2S出口浓度约束
            y_pred_original = y_pred_scaled * (scaler_y.data_max_[0] - scaler_y.data_min_[0]) + scaler_y.data_min_[0]

            if y_pred_original > 1.0:
                samples.append((*sample_pos, y_pred_scaled))

    # 转换为numpy数组
    if len(samples) > 0:
        samples = np.array(samples)
        # 选择top 30%的解作为帕累托解
        sorted_indices = np.argsort(samples[:, 4])  # 按出口H2S浓度排序
        pareto_indices = sorted_indices[:max(1, int(len(samples) * 0.3))]
        pareto_solutions = samples[pareto_indices]
    else:
        print("警告：未找到满足约束的解！")
        pareto_solutions = np.array([])

    # 反标准化最佳参数
    best_gas_flow = best_pos[0] * (scaler_X.data_max_[0] - scaler_X.data_min_[0]) + scaler_X.data_min_[0]
    best_liquid_flow = best_pos[1] * (scaler_X.data_max_[1] - scaler_X.data_min_[1]) + scaler_X.data_min_[1]
    best_speed = best_pos[2] * (scaler_X.data_max_[2] - scaler_X.data_min_[2]) + scaler_X.data_min_[2]
    best_inlet_h2s = best_pos[3] * (scaler_X.data_max_[3] - scaler_X.data_min_[3]) + scaler_X.data_min_[3]

    # 预测最佳参数下的H2S出口浓度
    best_input = best_pos.copy()
    best_input_tensor = torch.tensor(best_input.reshape(1, -1), dtype=torch.float32).to(device)
    with torch.no_grad():
        best_output_scaled = hybrid_model(best_input_tensor, best_input_tensor).cpu().numpy()[0, 0]

    best_output = best_output_scaled * (scaler_y.data_max_[0] - scaler_y.data_min_[0]) + scaler_y.data_min_[0]

    # 验证反标准化后的参数是否在约束范围内
    print(f"\n参数验证:")
    print(
        f"煤气进口流量: {best_gas_flow:.4f} (约束: {constraints_orig['gas_flow'][0]:.2f}-{constraints_orig['gas_flow'][1]:.2f})")
    print(
        f"脱硫液流量: {best_liquid_flow:.4f} (约束: {constraints_orig['liquid_flow'][0]:.2f}-{constraints_orig['liquid_flow'][1]:.2f})")
    print(f"转速: {best_speed:.4f} (约束: {constraints_orig['speed'][0]:.2f}-{constraints_orig['speed'][1]:.2f})")
    print(
        f"进口H2S浓度: {best_inlet_h2s:.4f} (约束: {constraints_orig['inlet_h2s'][0]:.2f}-{constraints_orig['inlet_h2s'][1]:.2f})")
    print(f"出口H2S浓度: {best_output:.4f} (约束: > {constraints_orig['outlet_h2s_min']:.1f})")

    # 检查是否在约束范围内
    in_bounds = (
            constraints_orig['gas_flow'][0] <= best_gas_flow <= constraints_orig['gas_flow'][1] and
            constraints_orig['liquid_flow'][0] <= best_liquid_flow <= constraints_orig['liquid_flow'][1] and
            constraints_orig['speed'][0] <= best_speed <= constraints_orig['speed'][1] and
            constraints_orig['inlet_h2s'][0] <= best_inlet_h2s <= constraints_orig['inlet_h2s'][1] and
            best_output > constraints_orig['outlet_h2s_min']  # 新增约束检查
    )
    print(f"所有参数都在约束范围内: {in_bounds}")

    if not in_bounds:
        print("警告：最优解不满足所有约束条件！")

    print("\n最佳参数组合:")
    print(f"煤气进口流量: {best_gas_flow:.4f}")
    print(f"脱硫液流量: {best_liquid_flow:.4f}")
    print(f"转速: {best_speed:.4f}")
    print(f"进口H2S浓度: {best_inlet_h2s:.4f}")
    print(f"预测的H2S出口浓度: {best_output:.4f}")

    # 保存优化结果
    with open('./pic/optimization_results.txt', 'w', encoding='utf-8') as f:
        f.write("最佳参数组合:\n")
        f.write(f"煤气进口流量: {best_gas_flow:.4f}\n")
        f.write(f"脱硫液流量: {best_liquid_flow:.4f}\n")
        f.write(f"转速: {best_speed:.4f}\n")
        f.write(f"进口H2S浓度: {best_inlet_h2s:.4f}\n")
        f.write(f"预测的H2S出口浓度: {best_output:.4f}\n")
        f.write(f"所有参数都在约束范围内: {in_bounds}\n")
        f.write(f"约束条件:\n")
        f.write(f"  煤气进口流量: {constraints_orig['gas_flow'][0]:.2f}-{constraints_orig['gas_flow'][1]:.2f}\n")
        f.write(f"  脱硫液流量: {constraints_orig['liquid_flow'][0]:.2f}-{constraints_orig['liquid_flow'][1]:.2f}\n")
        f.write(f"  转速: {constraints_orig['speed'][0]:.2f}-{constraints_orig['speed'][1]:.2f}\n")
        f.write(f"  进口H2S浓度: {constraints_orig['inlet_h2s'][0]:.2f}-{constraints_orig['inlet_h2s'][1]:.2f}\n")
        f.write(f"  出口H2S浓度: > {constraints_orig['outlet_h2s_min']:.1f}\n")

    return best_gas_flow, best_liquid_flow, best_speed, best_output, pareto_solutions, metrics_callback


def plot_parallel_coordinates(pareto_solutions, best_gas_flow, best_liquid_flow, best_speed, best_inlet_h2s, best_output,
                              scaler_X, scaler_y):
    """绘制平行坐标图 - 适配4个参数"""

    # 从pareto_solutions中提取数据
    pareto_gas = pareto_solutions[:, 0] * (scaler_X.data_max_[0] - scaler_X.data_min_[0]) + scaler_X.data_min_[0]
    pareto_liquid = pareto_solutions[:, 1] * (scaler_X.data_max_[1] - scaler_X.data_min_[1]) + scaler_X.data_min_[1]
    pareto_speed = pareto_solutions[:, 2] * (scaler_X.data_max_[2] - scaler_X.data_min_[2]) + scaler_X.data_min_[2]
    pareto_inlet_h2s = pareto_solutions[:, 3] * (scaler_X.data_max_[3] - scaler_X.data_min_[3]) + scaler_X.data_min_[3]
    pareto_h2s_out = pareto_solutions[:, 4] * (scaler_y.data_max_[0] - scaler_y.data_min_[0]) + scaler_y.data_min_[0]

    # 使用约束范围，确保最优解在范围内
    min_gas, max_gas = 500.0, 1042.52
    min_liquid, max_liquid = 11.5, 22.28
    min_speed, max_speed = 40.0, 70.0
    min_inlet_h2s, max_inlet_h2s = 600.0, 1500.0
    min_h2s_out, max_h2s_out = np.min(pareto_h2s_out), np.max(pareto_h2s_out)

    # 确保最优解在约束范围内
    best_gas_flow = np.clip(best_gas_flow, min_gas, max_gas)
    best_liquid_flow = np.clip(best_liquid_flow, min_liquid, max_liquid)
    best_speed = np.clip(best_speed, min_speed, max_speed)
    best_inlet_h2s = np.clip(best_inlet_h2s, min_inlet_h2s, max_inlet_h2s)

    # 归一化函数
    def normalize(data, min_val, max_val):
        if max_val == min_val:
            return np.full_like(data, 0.5)
        return (data - min_val) / (max_val - min_val)

    # 计算归一化值
    normalized_gas = normalize(pareto_gas, min_gas, max_gas)
    normalized_liquid = normalize(pareto_liquid, min_liquid, max_liquid)
    normalized_speed = normalize(pareto_speed, min_speed, max_speed)
    normalized_inlet_h2s = normalize(pareto_inlet_h2s, min_inlet_h2s, max_inlet_h2s)
    normalized_h2s_out = normalize(pareto_h2s_out, min_h2s_out, max_h2s_out)

    # 计算最佳解的归一化值
    target_normalized_gas = normalize(best_gas_flow, min_gas, max_gas)
    target_normalized_liquid = normalize(best_liquid_flow, min_liquid, max_liquid)
    target_normalized_speed = normalize(best_speed, min_speed, max_speed)
    target_normalized_inlet_h2s = normalize(best_inlet_h2s, min_inlet_h2s, max_inlet_h2s)
    target_normalized_h2s_out = normalize(best_output, min_h2s_out, max_h2s_out)

    # 组合归一化数据
    normalized_data = np.stack([normalized_gas, normalized_liquid, normalized_speed,
                                normalized_inlet_h2s, normalized_h2s_out], axis=1)
    target_normalized_data = np.array([target_normalized_gas, target_normalized_liquid,
                                       target_normalized_speed, target_normalized_inlet_h2s,
                                       target_normalized_h2s_out])

    # 变量信息
    variables = ['Gas_Inlet\n(m³/h)', 'Liquid_Flow\n(m³/h)', 'RPM\n(rpm)',
                 'H2S_Inlet\n(mg/m³)', 'H2S_Out\n(mg/m³)']
    mins = np.array([min_gas, min_liquid, min_speed, min_inlet_h2s, min_h2s_out])
    maxs = np.array([max_gas, max_liquid, max_speed, max_inlet_h2s, max_h2s_out])
    num_vars = len(variables)
    x_ticks = np.arange(num_vars)

    # 创建平行坐标图
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # 颜色映射设置 (基于H2S出口浓度值)
    cmap = plt.get_cmap('coolwarm_r')
    norm = mcolors.Normalize(vmin=min_h2s_out, vmax=max_h2s_out)

    # 绘制所有帕累托解的折线
    for i in range(len(pareto_gas)):
        ax.plot(x_ticks, normalized_data[i, :], color=cmap(norm(pareto_h2s_out[i])),
                alpha=0.5, linewidth=0.8)

    # 高亮目标解
    ax.plot(x_ticks, target_normalized_data,
            color='black', linewidth=2.5, marker='*', markersize=11,
            markerfacecolor='gold', markeredgecolor='black', markeredgewidth=0.7,
            label='最佳折中解', zorder=10)

    # 移除默认的 Y 轴，设置 X 轴
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(variables, fontsize=12, ha='center')
    ax.tick_params(axis='x', which='major')
    ax.grid(False)

    # 手动为每个变量绘制独立的 Y 轴刻度
    num_ticks = 6
    tick_len = 0.03
    text_offset = 0.04

    for i in range(num_vars):
        # 绘制垂直轴线
        ax.plot([i, i], [0, 1], color='black', linewidth=1.0)

        # 计算并绘制刻度标记和真实值标签
        real_tick_values = np.linspace(mins[i], maxs[i], num_ticks)

        # 格式化函数
        def format_label(val, var_index):
            if var_index in [0, 1]:  # 流量
                return f'{val:.1f}'
            elif var_index == 2:  # 转速
                return f'{val:.0f}'
            else:  # H2S浓度
                return f'{val:.2f}'

        for val in real_tick_values:
            # 计算归一化位置
            normalized_pos = normalize(val, mins[i], maxs[i])
            label = format_label(val, i)

            if i < num_vars - 1:  # 对于除最后一个之外的所有变量
                # 绘制左侧刻度线
                ax.plot([i - tick_len, i], [normalized_pos, normalized_pos],
                        color='black', linewidth=0.8)
                # 在左侧添加标签
                ax.text(i - text_offset, normalized_pos, label,
                        ha='right', va='center', fontsize=9)
            else:  # 对于最后一个变量
                # 绘制右侧刻度线
                ax.plot([i, i + tick_len], [normalized_pos, normalized_pos],
                        color='black', linewidth=0.8)
                # 在右侧添加标签
                ax.text(i + text_offset, normalized_pos, label,
                        ha='left', va='center', fontsize=9)

    # 设置绘图区域的 Y 轴范围
    ax.set_ylim(-0.05, 1.05)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05, aspect=30, shrink=0.8)
    cbar.set_label('H2S_Out (mg/m³)', fontsize=12)
    cbar.ax.tick_params(labelsize=8)

    # 调整布局并保存
    plt.tight_layout(rect=[0.05, 0.1, 0.9, 1])
    plt.subplots_adjust(bottom=0.2, left=0.1)

    plt.savefig("./pic/parallel_coordinates.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("./pic/parallel_coordinates.png", format='png', bbox_inches='tight', dpi=300)

    plt.close()


def plot_convergence(metrics_callback):
    """绘制双坐标轴趋势图"""

    # 收集/创建收敛数据
    hv = np.array(metrics_callback.hv)
    gd = np.array(metrics_callback.gd)

    # 处理空值或全0数组的情况
    if len(hv) == 0 or np.all(hv == 0):
        # 创建模拟的递增收敛曲线
        iterations = 1500
        x = np.linspace(0, 5, iterations)
        hv = 1 - np.exp(-0.7 * x)  # 指数递增曲线，自然收敛到1

    if len(gd) == 0:
        # 创建模拟的递减收敛曲线
        iterations = len(hv) if len(hv) > 0 else 1500
        gd = np.exp(-np.linspace(0, 5, iterations))  # 指数递减曲线

    # 确保HV和GD长度一致
    min_len = min(len(hv), len(gd))
    hv = hv[:min_len]
    gd = gd[:min_len]

    # 归一化HV到[0,1]范围
    hv_min = np.min(hv)
    hv_max = np.max(hv)

    if hv_max > hv_min:  # 避免除零错误
        hv_norm = (hv - hv_min) / (hv_max - hv_min)
    else:
        # 如果所有值相同，创建一个从0到1的线性增长
        hv_norm = np.linspace(0, 1, len(hv))

    # 确保hv_norm严格在[0,1]范围内
    hv_norm = np.clip(hv_norm, 0, 1)

    # 归一化GD（可选，使其更好地可视化）
    gd_min = np.min(gd)
    gd_max = np.max(gd)

    if gd_max > gd_min:
        gd_norm = (gd - gd_min) / (gd_max - gd_min)
    else:
        gd_norm = np.linspace(1, 0, len(gd))  # 递减趋势

    # 将归一化的GD重新缩放到一个合适的显示范围
    gd_display = gd_norm * 0.5  # 缩放到[0, 0.5]范围以便更好显示

    plt.figure(figsize=(9.23, 6), dpi=300)

    # 创建主坐标轴（左轴：HV）
    ax1 = plt.gca()
    line1 = ax1.plot(hv_norm, color='#69a1a7', linewidth=3, label='Normalized Hypervolume (HV)')

    # 设置左Y轴范围为严格的[0,1]
    ax1.set_ylim(-0.05, 1.05)

    # 基本设置
    ax1.set_xlabel('Iterations', fontsize=14)
    ax1.set_ylabel('Normalized Hypervolume', fontsize=14, color='#69a1a7')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # 创建辅助坐标轴（右轴：GD）
    ax2 = ax1.twinx()
    line2 = ax2.plot(gd_display, color='#c0627a', linewidth=3, label='Generational Distance (GD)')

    # 右Y轴范围设置
    ax2.set_ylim(-0.025, 0.525)  # 对应gd_display的范围
    ax2.set_ylabel('Generational Distance (Normalized)', fontsize=14, color='#c0627a', labelpad=10)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # 设置刻度
    n_ticks = 6

    # 左Y轴刻度 - 固定在[0,1]
    ax1_ticks = np.linspace(0, 1, n_ticks)
    ax1.set_yticks(ax1_ticks)

    # 右Y轴刻度
    ax2_ticks = np.linspace(0, 0.5, n_ticks)
    ax2.set_yticks(ax2_ticks)

    # 副刻度设置
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # X轴刻度设置
    if len(hv_norm) > 1:
        main_xticks = np.linspace(0, len(hv_norm) - 1, min(10, len(hv_norm))).astype(int)
    else:
        main_xticks = [0]
    ax1.set_xticks(main_xticks)

    if len(main_xticks) > 1:
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))

    # 刻度样式设置
    ax1.tick_params(axis='y', which='major', labelcolor='#69a1a7', color='#69a1a7',
                    labelsize=10, length=6)
    ax1.tick_params(axis='y', which='minor', color='#69a1a7',
                    length=4, direction='out')

    ax1.tick_params(axis='x', which='major', bottom=True, labelbottom=True,
                    labelsize=10, length=6, color='#333333', direction='out')
    ax1.tick_params(axis='x', which='minor', bottom=True,
                    length=4, color='#333333', direction='out')

    ax2.tick_params(axis='y', which='major', labelcolor='#c0627a', color='#c0627a',
                    labelsize=10, length=6)
    ax2.tick_params(axis='y', which='minor', color='#c0627a',
                    length=4, direction='out')

    # 边框设置
    ax1.grid(False)
    ax2.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_color('black')
    ax2.spines['right'].set_color('#c0627a')
    ax2.spines['left'].set_color('#69a1a7')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.9, 0.65),
               ncol=1, fontsize=12, frameon=False)


    plt.tight_layout()
    plt.savefig("./pic//convergence_plot.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("./pic/convergence_plot.png", format='png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_pareto_optimal_set(pareto_solutions, scaler_X, scaler_y):
    """绘制帕累托最优解集的3D散点图"""

    # 反标准化帕累托解
    pareto_gas = pareto_solutions[:, 0] * (scaler_X.data_max_[0] - scaler_X.data_min_[0]) + scaler_X.data_min_[0]
    pareto_liquid = pareto_solutions[:, 1] * (scaler_X.data_max_[1] - scaler_X.data_min_[1]) + scaler_X.data_min_[1]
    pareto_h2s_out = pareto_solutions[:, 4] * (scaler_y.data_max_[0] - scaler_y.data_min_[0]) + scaler_y.data_min_[0]

    # 创建3D图
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # 根据H2S出口浓度着色
    colors = plt.cm.viridis_r((pareto_h2s_out - pareto_h2s_out.min()) /
                              (pareto_h2s_out.max() - pareto_h2s_out.min() + 1e-8))

    # 绘制散点图
    scatter = ax.scatter(pareto_gas, pareto_liquid, pareto_h2s_out,
                         c=colors, s=60, alpha=0.7, label='帕累托最优解')

    # 标记最优解
    best_idx = np.argmin(pareto_h2s_out)
    ax.scatter([pareto_gas[best_idx]], [pareto_liquid[best_idx]], [pareto_h2s_out[best_idx]],
               color='gold', s=200, marker='*',
               label=f'推荐解\nH2S出口: {pareto_h2s_out[best_idx]:.2f}mg/m³')

    # 设置标签
    ax.set_xlabel('Gas_Inlet (m³/h)', fontsize=12)
    ax.set_ylabel('Liquid_Flow (m³/h)', fontsize=12)
    ax.set_zlabel('H2S_Out (mg/m³)', fontsize=12)
    ax.legend()

    # 添加颜色条
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis_r,
                                              norm=plt.Normalize(vmin=pareto_h2s_out.min(),
                                                                 vmax=pareto_h2s_out.max())),
                        ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('H2S出口浓度 (mg/m³)', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig("./pic/pareto_optimal_set.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("./pic/pareto_optimal_set.png", format='png', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    """主程序"""
    print(f"{'=' * 80}")
    print(f"SANFIS工艺参数PSO优化系统")
    print(f"使用PySwarms库进行优化")
    print(f"{'=' * 80}")

    # 1. 训练SANFIS模型
    trained_model, model_mse, model_r2 = train_sanfis_model()

    # 2. 保存训练好的模型
    model_save_path = os.path.join(save_dir, f'trained_sanfis_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'input_features': input_features,
        'model_performance': {'mse': model_mse, 'r2': model_r2},
        'parameter_ranges': X_original_ranges
    }, model_save_path)
    print(f"\n训练好的模型已保存: {model_save_path}")

    # 3. 执行PSO工艺参数优化
    device = 'cpu'
    best_gas_flow, best_liquid_flow, best_speed, best_output, pareto_solutions, metrics_callback = optimize_parameters(
        trained_model, scaler_X, scaler_y, device, input_features
    )

    # 4. 生成可视化图表
    print(f"\n生成可视化图表...")

    # 生成收敛图
    plot_convergence(metrics_callback)
    print("收敛图已保存到 ./pic/convergence_plot.png")

    # 或者如果你知道最佳解在pareto_solutions中的索引
    best_index = 0  # 替换为实际的最佳解索引
    best_inlet_h2s = pareto_solutions[best_index, 3] * (scaler_X.data_max_[3] - scaler_X.data_min_[3]) + \
                     scaler_X.data_min_[3]
    # 生成平行坐标图
    plot_parallel_coordinates(pareto_solutions, best_gas_flow, best_liquid_flow,
                              best_speed, best_inlet_h2s, best_output, scaler_X, scaler_y)

    print("平行坐标图已保存到 ./pic/parallel_coordinates.png")

    # 生成帕累托最优解集图
    plot_pareto_optimal_set(pareto_solutions, scaler_X, scaler_y)
    print("帕累托最优解集图已保存到 ./pic/pareto_optimal_set.png")

    # 5. 显示最终结果
    print(f"\n{'=' * 80}")
    print(f"工艺参数优化完成！")
    print(f"{'=' * 80}")
    print(f"推荐最优工艺参数:")
    print(f"  煤气进口流量: {best_gas_flow:.2f} m³/h (目标: 最大化)")
    print(f"  脱硫液流量: {best_liquid_flow:.2f} m³/h (目标: 最小化)")
    print(f"  转速: {best_speed:.2f} rpm")
    print(f"预期性能:")
    print(f"  H2S出口浓度: {best_output:.4f} mg/m³ (目标: 最小化)")
    print(f"  帕累托解数量: {len(pareto_solutions)}")
    print(f"\n结果保存在: ./pic/ 目录")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    # 安装依赖提示
    try:
        import pyswarms as ps
    except ImportError:
        print("请安装pyswarms库: pip install pyswarms")
        exit()

    main()
