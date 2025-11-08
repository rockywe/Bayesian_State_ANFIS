import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
save_dir = 'sanfis_process_optimization'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# [数据加载和预处理代码 - 与之前相同]
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
    'Gas_Inlet_Flow',  # 煤气进口流量 - 希望最大化
    'Desulfurization_Liquid_Flow',  # 脱硫液流量 - 希望最小化
    'Rotation_Speed',  # 转速
    'H2S_Inlet_Concentration'  # 进口H2S浓度
]
output_feature = 'H2S_Outlet_Concentration'  # 出口H2S浓度 - 希望最小化

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

# 获取原始数据的范围，用于PSO优化的边界
X_original_ranges = {
    'Gas_Inlet_Flow': (data_clean['Gas_Inlet_Flow'].min(), data_clean['Gas_Inlet_Flow'].max()),
    'Desulfurization_Liquid_Flow': (
    data_clean['Desulfurization_Liquid_Flow'].min(), data_clean['Desulfurization_Liquid_Flow'].max()),
    'Rotation_Speed': (data_clean['Rotation_Speed'].min(), data_clean['Rotation_Speed'].max()),
    'H2S_Inlet_Concentration': (
    data_clean['H2S_Inlet_Concentration'].min(), data_clean['H2S_Inlet_Concentration'].max())
}

print("原始数据范围:")
for feature, (min_val, max_val) in X_original_ranges.items():
    print(f"  {feature}: [{min_val:.2f}, {max_val:.2f}]")


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


class ProcessOptimizationPSO:
    """工艺参数优化PSO算法"""

    def __init__(self, trained_model, scaler_X, scaler_y, n_particles=50, n_iterations=200):
        self.model = trained_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.n_particles = n_particles
        self.n_iterations = n_iterations

        # PSO参数
        self.w = 0.9  # 惯性权重
        self.c1 = 2.0  # 个体学习因子
        self.c2 = 2.0  # 社会学习因子

        # 优化目标权重
        self.w_outlet_h2s = 0.5  # 出口H2S浓度权重（最小化）
        self.w_liquid_flow = 0.3  # 脱硫液流量权重（最小化）
        self.w_gas_flow = 0.2  # 煤气进口流量权重（最大化）

        # 约束条件
        self.max_outlet_h2s = 50.0  # 最大允许出口H2S浓度 (mg/m³)

        # 存储优化历史
        self.convergence_history = []
        self.pareto_solutions = []
        self.all_solutions = []

    def evaluate_process_objectives(self, params_original):
        """评估工艺目标函数"""
        try:
            # 标准化输入参数
            params_scaled = self.scaler_X.transform(params_original.reshape(1, -1))
            params_tensor = torch.tensor(params_scaled, dtype=torch.float32)

            # 使用训练好的模型预测
            self.model.eval()
            with torch.no_grad():
                outlet_h2s_scaled = self.model(params_tensor, params_tensor)
                outlet_h2s_original = self.scaler_y.inverse_transform(
                    outlet_h2s_scaled.detach().numpy())[0, 0]

            # 提取参数
            gas_inlet_flow = params_original[0]
            liquid_flow = params_original[1]
            rotation_speed = params_original[2]
            inlet_h2s = params_original[3]

            # 计算目标函数
            # 目标1: 最小化出口H2S浓度
            obj1 = outlet_h2s_original

            # 目标2: 最小化脱硫液流量
            obj2 = liquid_flow

            # 目标3: 最大化煤气进口流量 (转换为最小化问题)
            obj3 = -gas_inlet_flow

            # 约束惩罚
            penalty = 0
            if outlet_h2s_original > self.max_outlet_h2s:
                penalty += 1000 * (outlet_h2s_original - self.max_outlet_h2s)

            # 综合适应度函数 (加权和)
            fitness = (self.w_outlet_h2s * obj1 +
                       self.w_liquid_flow * obj2 +
                       self.w_gas_flow * obj3 + penalty)

            return fitness, obj1, obj2, obj3, outlet_h2s_original, penalty

        except Exception as e:
            print(f"评估目标函数时出错: {e}")
            return float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')

    def initialize_particles(self):
        """初始化粒子群"""
        particles = []
        velocities = []

        for i in range(self.n_particles):
            particle = []
            velocity = []

            # 煤气进口流量 (希望较高)
            gas_flow = np.random.uniform(
                X_original_ranges['Gas_Inlet_Flow'][0],
                X_original_ranges['Gas_Inlet_Flow'][1]
            )
            particle.append(gas_flow)
            velocity.append(np.random.uniform(-10, 10))

            # 脱硫液流量 (希望较低)
            liquid_flow = np.random.uniform(
                X_original_ranges['Desulfurization_Liquid_Flow'][0],
                X_original_ranges['Desulfurization_Liquid_Flow'][1]
            )
            particle.append(liquid_flow)
            velocity.append(np.random.uniform(-5, 5))

            # 转速
            rotation_speed = np.random.uniform(
                X_original_ranges['Rotation_Speed'][0],
                X_original_ranges['Rotation_Speed'][1]
            )
            particle.append(rotation_speed)
            velocity.append(np.random.uniform(-50, 50))

            # 进口H2S浓度 (通常由上游工艺决定，范围相对固定)
            inlet_h2s = np.random.uniform(
                X_original_ranges['H2S_Inlet_Concentration'][0],
                X_original_ranges['H2S_Inlet_Concentration'][1]
            )
            particle.append(inlet_h2s)
            velocity.append(np.random.uniform(-10, 10))

            particles.append(np.array(particle))
            velocities.append(np.array(velocity))

        return np.array(particles), np.array(velocities)

    def update_particles(self, particles, velocities, pbest, gbest, iteration):
        """更新粒子位置和速度"""
        # 动态调整惯性权重
        w_current = self.w * (1 - iteration / self.n_iterations)

        for i in range(self.n_particles):
            r1 = np.random.random(particles[i].shape)
            r2 = np.random.random(particles[i].shape)

            # 更新速度
            velocities[i] = (w_current * velocities[i] +
                             self.c1 * r1 * (pbest[i] - particles[i]) +
                             self.c2 * r2 * (gbest - particles[i]))

            # 速度限制
            velocities[i] = np.clip(velocities[i], -50, 50)

            # 更新位置
            particles[i] += velocities[i]

            # 边界约束
            particles[i][0] = np.clip(particles[i][0],
                                      X_original_ranges['Gas_Inlet_Flow'][0],
                                      X_original_ranges['Gas_Inlet_Flow'][1])
            particles[i][1] = np.clip(particles[i][1],
                                      X_original_ranges['Desulfurization_Liquid_Flow'][0],
                                      X_original_ranges['Desulfurization_Liquid_Flow'][1])
            particles[i][2] = np.clip(particles[i][2],
                                      X_original_ranges['Rotation_Speed'][0],
                                      X_original_ranges['Rotation_Speed'][1])
            particles[i][3] = np.clip(particles[i][3],
                                      X_original_ranges['H2S_Inlet_Concentration'][0],
                                      X_original_ranges['H2S_Inlet_Concentration'][1])

        return particles, velocities

    def optimize(self):
        """执行PSO优化"""
        print(f"\n{'=' * 60}")
        print(f"开始工艺参数PSO优化")
        print(f"粒子数: {self.n_particles}, 迭代次数: {self.n_iterations}")
        print(f"{'=' * 60}")

        # 初始化粒子群
        particles, velocities = self.initialize_particles()

        # 初始化最优解
        pbest = particles.copy()
        pbest_fitness = [float('inf')] * self.n_particles
        gbest = None
        gbest_fitness = float('inf')
        gbest_details = None

        for iteration in range(self.n_iterations):
            print(f"\n迭代 {iteration + 1}/{self.n_iterations}")

            iteration_fitness = []
            iteration_details = []

            for i in range(self.n_particles):
                # 评估适应度
                fitness, obj1, obj2, obj3, outlet_h2s, penalty = self.evaluate_process_objectives(particles[i])

                iteration_fitness.append(fitness)

                # 存储详细信息
                details = {
                    'particle_id': i,
                    'iteration': iteration,
                    'params': particles[i].copy(),
                    'gas_inlet_flow': particles[i][0],
                    'liquid_flow': particles[i][1],
                    'rotation_speed': particles[i][2],
                    'inlet_h2s': particles[i][3],
                    'outlet_h2s': outlet_h2s,
                    'fitness': fitness,
                    'obj1_outlet_h2s': obj1,
                    'obj2_liquid_flow': obj2,
                    'obj3_gas_flow': obj3,
                    'penalty': penalty,
                    'feasible': penalty == 0
                }

                iteration_details.append(details)
                self.all_solutions.append(details)

                # 更新个体最优
                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest[i] = particles[i].copy()

                # 更新全局最优
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest = particles[i].copy()
                    gbest_details = details.copy()

            # 记录收敛历史
            feasible_solutions = [d for d in iteration_details if d['feasible']]

            self.convergence_history.append({
                'iteration': iteration,
                'best_fitness': gbest_fitness,
                'avg_fitness': np.mean(iteration_fitness),
                'worst_fitness': np.max(iteration_fitness),
                'feasible_count': len(feasible_solutions),
                'best_outlet_h2s': gbest_details['outlet_h2s'] if gbest_details else float('inf'),
                'best_liquid_flow': gbest_details['liquid_flow'] if gbest_details else float('inf'),
                'best_gas_flow': gbest_details['gas_inlet_flow'] if gbest_details else 0
            })

            # 更新粒子
            particles, velocities = self.update_particles(particles, velocities, pbest, gbest, iteration)

            # 打印进度
            if gbest_details:
                print(f"  最佳适应度: {gbest_fitness:.4f}")
                print(f"  出口H2S: {gbest_details['outlet_h2s']:.2f} mg/m³")
                print(f"  脱硫液流量: {gbest_details['liquid_flow']:.2f}")
                print(f"  煤气进口流量: {gbest_details['gas_inlet_flow']:.2f}")
                print(f"  可行解数量: {len(feasible_solutions)}/{self.n_particles}")

        # 找到帕累托最优解
        self.find_pareto_solutions()

        return gbest, gbest_fitness, gbest_details

    def find_pareto_solutions(self):
        """找到帕累托最优解"""
        feasible_solutions = [sol for sol in self.all_solutions if sol['feasible']]

        pareto_solutions = []

        for i, sol1 in enumerate(feasible_solutions):
            is_pareto = True

            for j, sol2 in enumerate(feasible_solutions):
                if i != j:
                    # 检查sol2是否支配sol1
                    # 支配条件：在所有目标上不差于sol1，且至少在一个目标上更好
                    if (sol2['obj1_outlet_h2s'] <= sol1['obj1_outlet_h2s'] and
                            sol2['obj2_liquid_flow'] <= sol1['obj2_liquid_flow'] and
                            sol2['obj3_gas_flow'] <= sol1['obj3_gas_flow'] and
                            (sol2['obj1_outlet_h2s'] < sol1['obj1_outlet_h2s'] or
                             sol2['obj2_liquid_flow'] < sol1['obj2_liquid_flow'] or
                             sol2['obj3_gas_flow'] < sol1['obj3_gas_flow'])):
                        is_pareto = False
                        break

            if is_pareto:
                pareto_solutions.append(sol1)

        self.pareto_solutions = pareto_solutions
        print(f"\n找到 {len(pareto_solutions)} 个帕累托最优解")


def plot_convergence(pso_optimizer, save_dir):
    """绘制收敛曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    iterations = [h['iteration'] for h in pso_optimizer.convergence_history]
    best_fitness = [h['best_fitness'] for h in pso_optimizer.convergence_history]
    avg_fitness = [h['avg_fitness'] for h in pso_optimizer.convergence_history]
    feasible_count = [h['feasible_count'] for h in pso_optimizer.convergence_history]

    # 适应度收敛
    ax1.plot(iterations, best_fitness, 'b-', linewidth=3, label='最佳适应度')
    ax1.plot(iterations, avg_fitness, 'g-', linewidth=2, label='平均适应度')
    ax1.set_title('适应度收敛曲线', fontweight='bold', fontsize=14)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('适应度值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 出口H2S浓度收敛
    outlet_h2s = [h['best_outlet_h2s'] for h in pso_optimizer.convergence_history]
    ax2.plot(iterations, outlet_h2s, 'r-', linewidth=3)
    ax2.axhline(y=pso_optimizer.max_outlet_h2s, color='red', linestyle='--',
                label=f'约束上限 ({pso_optimizer.max_outlet_h2s} mg/m³)')
    ax2.set_title('出口H2S浓度优化', fontweight='bold', fontsize=14)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('出口H2S浓度 (mg/m³)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 脱硫液流量优化
    liquid_flow = [h['best_liquid_flow'] for h in pso_optimizer.convergence_history]
    ax3.plot(iterations, liquid_flow, 'purple', linewidth=3)
    ax3.set_title('脱硫液流量优化 (最小化)', fontweight='bold', fontsize=14)
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('脱硫液流量')
    ax3.grid(True, alpha=0.3)

    # 可行解数量
    ax4.plot(iterations, feasible_count, 'orange', linewidth=3, marker='o')
    ax4.set_title('可行解数量变化', fontweight='bold', fontsize=14)
    ax4.set_xlabel('迭代次数')
    ax4.set_ylabel('可行解数量')
    ax4.set_ylim(0, pso_optimizer.n_particles + 5)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'process_optimization_convergence_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_parallel_coordinates(pso_optimizer, save_dir):
    """绘制工艺参数平行坐标图"""
    pareto_solutions = pso_optimizer.pareto_solutions

    if not pareto_solutions:
        print("没有找到帕累托最优解")
        return

    # 准备数据
    data = []
    for sol in pareto_solutions:
        row = [
            sol['gas_inlet_flow'],  # 煤气进口流量
            sol['liquid_flow'],  # 脱硫液流量
            sol['rotation_speed'],  # 转速
            sol['inlet_h2s'],  # 进口H2S浓度
            sol['outlet_h2s'],  # 出口H2S浓度
            sol['fitness']  # 适应度
        ]
        data.append(row)

    data = np.array(data)

    # 参数名称
    param_names = [
        '煤气进口流量',
        '脱硫液流量',
        '转速',
        '进口H2S浓度',
        '出口H2S浓度',
        '综合适应度'
    ]

    # 标准化数据
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)

    # 绘制平行坐标图
    fig, ax = plt.subplots(figsize=(16, 10))

    # 根据综合适应度着色
    fitness_values = data[:, -1]
    colors = plt.cm.viridis_r((fitness_values - fitness_values.min()) /
                              (fitness_values.max() - fitness_values.min() + 1e-8))

    # 绘制每一条线
    for i in range(len(data_norm)):
        ax.plot(range(len(param_names)), data_norm[i],
                color=colors[i], alpha=0.7, linewidth=2)

    # 标记最优解
    best_idx = np.argmin(fitness_values)
    ax.plot(range(len(param_names)), data_norm[best_idx],
            color='gold', linewidth=4, marker='*', markersize=12,
            label=f'最优解 (适应度={fitness_values[best_idx]:.2f})')

    # 设置坐标轴
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_ylabel('标准化参数值')
    ax.set_title('工艺参数帕累托最优解平行坐标图', fontweight='bold', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r,
                               norm=plt.Normalize(vmin=fitness_values.min(),
                                                  vmax=fitness_values.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('综合适应度 (越小越好)', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'process_parallel_coordinates_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_pareto_optimal_3d(pso_optimizer, save_dir):
    """绘制3D帕累托最优解"""
    pareto_solutions = pso_optimizer.pareto_solutions

    if not pareto_solutions:
        print("没有找到帕累托最优解")
        return

    # 提取数据
    outlet_h2s = [sol['outlet_h2s'] for sol in pareto_solutions]
    liquid_flow = [sol['liquid_flow'] for sol in pareto_solutions]
    gas_flow = [sol['gas_inlet_flow'] for sol in pareto_solutions]
    fitness_values = [sol['fitness'] for sol in pareto_solutions]

    # 创建3D图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 根据综合适应度着色
    colors = plt.cm.viridis_r((np.array(fitness_values) - min(fitness_values)) /
                              (max(fitness_values) - min(fitness_values) + 1e-8))

    # 绘制帕累托前沿
    scatter = ax.scatter(outlet_h2s, liquid_flow, gas_flow,
                         c=colors, s=80, alpha=0.8,
                         label='帕累托最优解')

    # 标记最优解
    best_idx = np.argmin(fitness_values)
    ax.scatter([outlet_h2s[best_idx]], [liquid_flow[best_idx]], [gas_flow[best_idx]],
               color='gold', s=300, marker='*',
               label=f'推荐最优解\n出口H2S: {outlet_h2s[best_idx]:.1f}mg/m³')

    # 设置标签和标题
    ax.set_xlabel('出口H2S浓度 (mg/m³)')
    ax.set_ylabel('脱硫液流量')
    ax.set_zlabel('煤气进口流量')
    ax.set_title('工艺参数帕累托最优解集\n(低出口浓度 + 低脱硫液流量 + 高煤气流量)',
                 fontweight='bold', fontsize=14)
    ax.legend()

    # 添加约束平面
    if pso_optimizer.max_outlet_h2s:
        xx, yy = np.meshgrid(
            np.linspace(min(liquid_flow), max(liquid_flow), 10),
            np.linspace(min(gas_flow), max(gas_flow), 10)
        )
        zz = np.full_like(xx, pso_optimizer.max_outlet_h2s)
        ax.plot_surface(zz, xx, yy, alpha=0.2, color='red',
                        label=f'约束边界 ({pso_optimizer.max_outlet_h2s} mg/m³)')

    # 添加颜色条
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis_r,
                                              norm=plt.Normalize(vmin=min(fitness_values),
                                                                 vmax=max(fitness_values))),
                        ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('综合适应度', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'process_pareto_3d_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def save_optimization_results(pso_optimizer, best_params, best_details, save_dir):
    """保存优化结果"""

    # 准备保存数据
    results = {
        'optimization_config': {
            'objective': '最小化出口H2S浓度 + 最小化脱硫液流量 + 最大化煤气进口流量',
            'constraints': f'出口H2S浓度 ≤ {pso_optimizer.max_outlet_h2s} mg/m³',
            'weights': {
                'outlet_h2s': pso_optimizer.w_outlet_h2s,
                'liquid_flow': pso_optimizer.w_liquid_flow,
                'gas_flow': pso_optimizer.w_gas_flow
            }
        },
        'best_solution': {
            'parameters': {
                'gas_inlet_flow': float(best_details['gas_inlet_flow']),
                'desulfurization_liquid_flow': float(best_details['liquid_flow']),
                'rotation_speed': float(best_details['rotation_speed']),
                'h2s_inlet_concentration': float(best_details['inlet_h2s'])
            },
            'performance': {
                'outlet_h2s_concentration': float(best_details['outlet_h2s']),
                'fitness': float(best_details['fitness']),
                'feasible': best_details['feasible']
            }
        },
        'pareto_solutions': [],
        'convergence_history': pso_optimizer.convergence_history,
        'timestamp': timestamp
    }

    # 添加帕累托最优解
    for sol in pso_optimizer.pareto_solutions[:20]:  # 保存前20个
        pareto_sol = {
            'parameters': {
                'gas_inlet_flow': float(sol['gas_inlet_flow']),
                'desulfurization_liquid_flow': float(sol['liquid_flow']),
                'rotation_speed': float(sol['rotation_speed']),
                'h2s_inlet_concentration': float(sol['inlet_h2s'])
            },
            'performance': {
                'outlet_h2s_concentration': float(sol['outlet_h2s']),
                'fitness': float(sol['fitness'])
            }
        }
        results['pareto_solutions'].append(pareto_sol)

    # 保存JSON文件
    results_file = os.path.join(save_dir, f'process_optimization_results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存可读报告
    report_file = os.path.join(save_dir, f'optimization_report_{timestamp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SANFIS工艺参数优化报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"优化时间: {timestamp}\n\n")

        f.write("优化目标:\n")
        f.write("  1. 最小化出口H2S浓度\n")
        f.write("  2. 最小化脱硫液流量\n")
        f.write("  3. 最大化煤气进口流量\n\n")

        f.write("约束条件:\n")
        f.write(f"  出口H2S浓度 ≤ {pso_optimizer.max_outlet_h2s} mg/m³\n\n")

        f.write("推荐最优工艺参数:\n")
        f.write(f"  煤气进口流量: {best_details['gas_inlet_flow']:.2f}\n")
        f.write(f"  脱硫液流量: {best_details['liquid_flow']:.2f}\n")
        f.write(f"  转速: {best_details['rotation_speed']:.2f}\n")
        f.write(f"  进口H2S浓度: {best_details['inlet_h2s']:.2f}\n\n")

        f.write("预期性能:\n")
        f.write(f"  出口H2S浓度: {best_details['outlet_h2s']:.2f} mg/m³\n")
        f.write(f"  综合适应度: {best_details['fitness']:.4f}\n")
        f.write(f"  满足约束: {'是' if best_details['feasible'] else '否'}\n\n")

        f.write(f"帕累托最优解数量: {len(pso_optimizer.pareto_solutions)}\n")

    print(f"优化结果已保存:")
    print(f"  JSON文件: {results_file}")
    print(f"  报告文件: {report_file}")


def main():
    """主程序"""
    print(f"{'=' * 80}")
    print(f"SANFIS工艺参数PSO优化系统")
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
        'model_performance': {'mse': model_mse, 'r2': model_r2}
    }, model_save_path)
    print(f"\n训练好的模型已保存: {model_save_path}")

    # 3. 创建优化目录
    opt_dir = os.path.join(save_dir, f'process_optimization_{timestamp}')
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)

    # 4. 执行PSO工艺参数优化
    pso_optimizer = ProcessOptimizationPSO(
        trained_model=trained_model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        n_particles=30,
        n_iterations=100
    )

    best_params, best_fitness, best_details = pso_optimizer.optimize()

    # 5. 生成可视化图表
    print(f"\n生成可视化图表...")
    plot_convergence(pso_optimizer, opt_dir)
    plot_parallel_coordinates(pso_optimizer, opt_dir)
    plot_pareto_optimal_3d(pso_optimizer, opt_dir)

    # 6. 保存优化结果
    save_optimization_results(pso_optimizer, best_params, best_details, opt_dir)

    # 7. 显示最终结果
    print(f"\n{'=' * 60}")
    print(f"工艺参数优化完成！")
    print(f"{'=' * 60}")
    print(f"推荐最优工艺参数:")
    print(f"  煤气进口流量: {best_details['gas_inlet_flow']:.2f}")
    print(f"  脱硫液流量: {best_details['liquid_flow']:.2f}")
    print(f"  转速: {best_details['rotation_speed']:.2f}")
    print(f"  进口H2S浓度: {best_details['inlet_h2s']:.2f}")
    print(f"\n预期性能:")
    print(f"  出口H2S浓度: {best_details['outlet_h2s']:.2f} mg/m³")
    print(f"  综合适应度: {best_details['fitness']:.4f}")
    print(f"  帕累托解数量: {len(pso_optimizer.pareto_solutions)}")
    print(f"\n结果保存在: {opt_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
