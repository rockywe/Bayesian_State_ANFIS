import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sanfis import SANFIS, plottingtools
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import copy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# [前面的数据加载和预处理代码保持不变...]
# 加载数据
file_path = './data/脱硫数据整理2.xlsx'
try:
    data_df = pd.read_excel(file_path, sheet_name='Sheet1')
except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径：{file_path}")
    exit()

# 数据预处理部分
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
    'Gas_Inlet_Flow',
    'Desulfurization_Liquid_Flow',
    'Rotation_Speed',
    'H2S_Inlet_Concentration'
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


# === 关键：保存训练前的参数 ===
def extract_membership_params(model):
    """提取隶属度函数参数"""
    params = {
        'mu_params': [],
        'sigma_params': []
    }

    # 遍历模型参数
    for name, param in model.named_parameters():
        if 'mu' in name.lower() or 'mean' in name.lower():
            params['mu_params'].append(param.detach().cpu().numpy().copy())
        elif 'sigma' in name.lower() or 'std' in name.lower():
            params['sigma_params'].append(param.detach().cpu().numpy().copy())

    return params


def extract_membership_params_alternative(model, membfuncs_config):
    """备用方法：从配置和模型状态提取参数"""
    params = {
        'mu_params': [],
        'sigma_params': []
    }

    # 如果直接提取失败，使用模型的state_dict
    state_dict = model.state_dict()

    for i in range(len(membfuncs_config)):
        mu_list = []
        sigma_list = []

        # 查找相关参数
        for key, value in state_dict.items():
            if f'{i}' in key or 'membfunc' in key:
                if 'mu' in key or 'mean' in key:
                    if isinstance(value, torch.Tensor):
                        mu_list.extend(value.detach().cpu().numpy().flatten())
                elif 'sigma' in key or 'std' in key:
                    if isinstance(value, torch.Tensor):
                        sigma_list.extend(value.detach().cpu().numpy().flatten())

        # 如果仍然为空，使用初始配置
        if not mu_list:
            mu_list = membfuncs_config[i]['params']['mu']['value']
        if not sigma_list:
            sigma_list = membfuncs_config[i]['params']['sigma']['value']

        params['mu_params'].append(np.array(mu_list))
        params['sigma_params'].append(np.array(sigma_list))

    return params


# 保存训练前的参数
print("保存训练前的隶属度函数参数...")
initial_params = extract_membership_params_alternative(model, membfuncs_config)
print(f"初始μ参数: {[arr.tolist() for arr in initial_params['mu_params']]}")
print(f"初始σ参数: {[arr.tolist() for arr in initial_params['sigma_params']]}")

# 训练过程中定期保存参数
training_history = {
    'epochs': [],
    'mu_params': [],
    'sigma_params': [],
    'train_losses': [],
    'valid_losses': []
}

# 训练设置
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 2000
print_interval = 20
save_interval = 40  # 每40个epoch保存一次参数

print(f"\n开始训练SANFIS模型...")

model.train()
for epoch in range(epochs):
    # 训练步骤
    optimizer.zero_grad()
    train_pred = model(train_data[0], train_data[1])
    train_loss = loss_function(train_pred, train_data[2])
    train_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # 验证步骤
    model.eval()
    with torch.no_grad():
        valid_pred = model(valid_data[0], valid_data[1])
        valid_loss = loss_function(valid_pred, valid_data[2])
    model.train()

    # 保存训练历史
    if epoch % save_interval == 0 or epoch == epochs - 1:
        current_params = extract_membership_params_alternative(model, membfuncs_config)
        training_history['epochs'].append(epoch)
        training_history['mu_params'].append(copy.deepcopy(current_params['mu_params']))
        training_history['sigma_params'].append(copy.deepcopy(current_params['sigma_params']))
        training_history['train_losses'].append(train_loss.item())
        training_history['valid_losses'].append(valid_loss.item())

    # 打印进度
    if (epoch + 1) % print_interval == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss.item():.6f}, "
              f"Valid Loss: {valid_loss.item():.6f}")

# 获取训练后的参数
final_params = extract_membership_params_alternative(model, membfuncs_config)
print(f"\n最终μ参数: {[arr.tolist() for arr in final_params['mu_params']]}")
print(f"最终σ参数: {[arr.tolist() for arr in final_params['sigma_params']]}")


# === 可视化训练前后的变化 ===
def plot_membership_evolution(initial_params, final_params, training_history, input_features):
    """绘制隶属度函数的演化过程"""

    n_inputs = len(input_features)
    n_points = 100
    x = np.linspace(0, 1, n_points)

    # 1. 训练前后对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for i in range(n_inputs):
        ax = axes[i]

        # 初始隶属函数
        mu_init = initial_params['mu_params'][i]
        sigma_init = initial_params['sigma_params'][i]

        # 最终隶属函数
        mu_final = final_params['mu_params'][i]
        sigma_final = final_params['sigma_params'][i]

        # 确保参数长度匹配
        n_funcs = min(len(mu_init), len(sigma_init), len(mu_final), len(sigma_final))

        for j in range(n_funcs):
            # 训练前的隶属函数
            if j < len(mu_init) and j < len(sigma_init):
                membership_init = np.exp(-0.5 * ((x - mu_init[j]) / sigma_init[j]) ** 2)
                ax.plot(x, membership_init, '--', linewidth=2, alpha=0.7,
                        label=f'初始MF{j + 1} (μ={mu_init[j]:.3f})')

            # 训练后的隶属函数
            if j < len(mu_final) and j < len(sigma_final):
                membership_final = np.exp(-0.5 * ((x - mu_final[j]) / sigma_final[j]) ** 2)
                ax.plot(x, membership_final, '-', linewidth=3,
                        label=f'最终MF{j + 1} (μ={mu_final[j]:.3f})')

        ax.set_title(f'{input_features[i]}的隶属函数变化', fontsize=12, fontweight='bold')
        ax.set_xlabel('标准化输入值')
        ax.set_ylabel('隶属度')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.1)

    plt.suptitle('SANFIS隶属度函数训练前后对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 2. 参数变化趋势图
    if len(training_history['epochs']) > 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        epochs = training_history['epochs']

        # μ参数变化
        for i in range(n_inputs):
            mu_evolution = []
            for epoch_params in training_history['mu_params']:
                if i < len(epoch_params) and len(epoch_params[i]) > 0:
                    mu_evolution.append(np.mean(epoch_params[i]))
                else:
                    mu_evolution.append(np.nan)

            if len(mu_evolution) == len(epochs):
                ax1.plot(epochs, mu_evolution, 'o-', label=f'{input_features[i]}', linewidth=2)

        ax1.set_title('μ参数变化趋势', fontweight='bold')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('平均μ值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # σ参数变化
        for i in range(n_inputs):
            sigma_evolution = []
            for epoch_params in training_history['sigma_params']:
                if i < len(epoch_params) and len(epoch_params[i]) > 0:
                    sigma_evolution.append(np.mean(epoch_params[i]))
                else:
                    sigma_evolution.append(np.nan)

            if len(sigma_evolution) == len(epochs):
                ax2.plot(epochs, sigma_evolution, 's-', label=f'{input_features[i]}', linewidth=2)

        ax2.set_title('σ参数变化趋势', fontweight='bold')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('平均σ值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 损失变化
        ax3.plot(epochs, training_history['train_losses'], 'b-', label='训练损失', linewidth=2)
        ax3.plot(epochs, training_history['valid_losses'], 'r-', label='验证损失', linewidth=2)
        ax3.set_title('损失函数变化', fontweight='bold')
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('损失值')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # 参数变化幅度
        param_changes = []
        for i in range(len(epochs)):
            if i == 0:
                param_changes.append(0)
            else:
                # 计算相对于初始值的变化
                total_change = 0
                count = 0
                for j in range(n_inputs):
                    if (j < len(training_history['mu_params'][i]) and
                            j < len(initial_params['mu_params']) and
                            len(training_history['mu_params'][i][j]) > 0 and
                            len(initial_params['mu_params'][j]) > 0):
                        current_mu = np.mean(training_history['mu_params'][i][j])
                        initial_mu = np.mean(initial_params['mu_params'][j])
                        change = abs(current_mu - initial_mu) / (abs(initial_mu) + 1e-8)
                        total_change += change
                        count += 1

                if count > 0:
                    param_changes.append(total_change / count)
                else:
                    param_changes.append(0)

        ax4.plot(epochs, param_changes, 'g-', linewidth=2, marker='o')
        ax4.set_title('参数变化幅度', fontweight='bold')
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('相对变化幅度')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def print_parameter_changes(initial_params, final_params, input_features):
    """打印参数变化的详细信息"""
    print(f"\n{'=' * 60}")
    print(f"隶属度函数参数变化详细分析")
    print(f"{'=' * 60}")

    for i, feature in enumerate(input_features):
        print(f"\n【{feature}】")

        if (i < len(initial_params['mu_params']) and
                i < len(final_params['mu_params']) and
                i < len(initial_params['sigma_params']) and
                i < len(final_params['sigma_params'])):

            mu_init = initial_params['mu_params'][i]
            mu_final = final_params['mu_params'][i]
            sigma_init = initial_params['sigma_params'][i]
            sigma_final = final_params['sigma_params'][i]

            n_funcs = min(len(mu_init), len(mu_final), len(sigma_init), len(sigma_final))

            for j in range(n_funcs):
                print(f"  隶属函数 {j + 1}:")
                print(f"    μ: {mu_init[j]:.4f} → {mu_final[j]:.4f} (变化: {mu_final[j] - mu_init[j]:+.4f})")
                print(
                    f"    σ: {sigma_init[j]:.4f} → {sigma_final[j]:.4f} (变化: {sigma_final[j] - sigma_init[j]:+.4f})")
        else:
            print("  参数提取失败")


# 执行可视化
print("\n" + "=" * 60)
print("开始可视化隶属度函数的训练演化过程")
print("=" * 60)

plot_membership_evolution(initial_params, final_params, training_history, input_features)
print_parameter_changes(initial_params, final_params, input_features)

# 继续原有的模型评估
model.eval()
with torch.no_grad():
    y_pred_scaled_tensor = model(S_test_tensor, X_test_tensor)

y_pred_scaled = y_pred_scaled_tensor.detach().numpy()
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_np)

mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print(f"\n模型评估结果:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 绘制预测结果对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred, alpha=0.6)
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title(f'SANFIS预测结果 (R² = {r2:.4f})')
plt.grid(True, alpha=0.3)
plt.show()

import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import product

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def debug_model_parameters(model):
    """详细分析模型参数结构"""
    print(f"\n{'=' * 80}")
    print(f"SANFIS模型参数详细分析")
    print(f"{'=' * 80}")

    print("所有模型参数:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} | requires_grad: {param.requires_grad}")
        if param.numel() < 50:  # 如果参数不太多，显示具体数值
            print(f"    数值: {param.detach().cpu().numpy().flatten()}")
        print()

    print("模型子模块:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            print(f"  {name}: {type(module).__name__}")
            if hasattr(module, 'weight') and module.weight is not None:
                print(f"    -> weight: {module.weight.shape}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"    -> bias: {module.bias.shape}")

    print("模型结构:")
    print(model)

# === 提取完整的TSK模糊规则 ===
def extract_complete_fuzzy_rules(model, input_features):
    """提取完整的TSK模糊规则，包括前件和后件"""
    print(f"\n{'=' * 80}")
    print(f"SANFIS TSK型模糊规则完整提取")
    print(f"{'=' * 80}")

    n_inputs = len(input_features)
    n_mf = n_memb_funcs_per_input

    # 生成所有规则组合
    rule_combinations = list(product(range(n_mf), repeat=n_inputs))
    total_rules = len(rule_combinations)

    print(f"规则结构: TSK型 (Takagi-Sugeno-Kang)")
    print(f"规则形式: IF (前件条件) THEN y = w0 + w1×x1 + w2×x2 + ... + wn×xn")
    print(f"输入变量数量: {n_inputs}")
    print(f"每个输入的隶属函数数量: {n_mf}")
    print(f"总规则数量: {total_rules}")

    # 提取模型参数
    model_params = {}
    for name, param in model.named_parameters():
        model_params[name] = param.detach().cpu().numpy()

    print(f"\n模型参数结构:")
    for name, param in model_params.items():
        print(f"  {name}: shape {param.shape}")

    # 提取隶属函数参数
    mu_params = []
    sigma_params = []

    for name, param in model_params.items():
        if 'mu' in name.lower() or 'mean' in name.lower():
            mu_params.append(param)
        elif 'sigma' in name.lower() or 'std' in name.lower():
            sigma_params.append(param)

    print(f"\n隶属函数参数数量:")
    print(f"  mu参数组: {len(mu_params)}")
    print(f"  sigma参数组: {len(sigma_params)}")

    # 提取后件线性参数 (权重)
    consequent_params = []
    for name, param in model_params.items():
        if any(keyword in name.lower() for keyword in ['weight', 'linear', 'consequent', 'coeff']):
            consequent_params.append((name, param))

    print(f"\n后件参数:")
    if len(consequent_params) > 0:
        for name, param in consequent_params:
            print(f"  {name}: shape {param.shape}")
    else:
        print("  警告: 未找到明确的后件参数!")
        print("  尝试使用所有权重参数...")
        for name, param in model_params.items():
            if 'weight' in name.lower():
                consequent_params.append((name, param))
                print(f"  发现权重参数: {name}: shape {param.shape}")

    # 定义隶属函数标签
    mf_labels = ['低', '中', '高']

    # 计算规则激活强度
    model.eval()
    rule_data = []

    with torch.no_grad():
        # 使用一些测试样本
        sample_size = min(20, len(X_test_tensor))
        X_sample = X_test_tensor[:sample_size]
        S_sample = S_test_tensor[:sample_size]

        for rule_idx, rule_combo in enumerate(rule_combinations):
            # 计算前件激活强度
            total_activation = 0

            for sample_idx in range(len(X_sample)):
                rule_activation = 1.0

                for input_idx, mf_idx in enumerate(rule_combo):
                    x_val = X_sample[sample_idx, input_idx].item()

                    # 使用提取的隶属函数参数
                    if input_idx < len(mu_params) and input_idx < len(sigma_params):
                        mu_vals = mu_params[input_idx].flatten()
                        sigma_vals = sigma_params[input_idx].flatten()

                        if mf_idx < len(mu_vals) and mf_idx < len(sigma_vals):
                            mu = mu_vals[mf_idx]
                            sigma = sigma_vals[mf_idx]
                            membership = np.exp(-0.5 * ((x_val - mu) / (sigma + 1e-8)) ** 2)
                            rule_activation *= membership

                total_activation += rule_activation

            avg_activation = total_activation / len(X_sample)

            # 提取该规则的后件参数
            consequent_weights = None
            if len(consequent_params) > 0:
                param_name, param_values = consequent_params[0]

                try:
                    if param_values.ndim >= 2:
                        if rule_idx < param_values.shape[0]:
                            consequent_weights = param_values[rule_idx]
                        elif param_values.shape[1] >= n_inputs + 1:
                            flat_idx = rule_idx
                            if flat_idx < param_values.shape[0]:
                                consequent_weights = param_values[flat_idx]
                    elif param_values.ndim == 1:
                        params_per_rule = n_inputs + 1
                        start_idx = rule_idx * params_per_rule
                        end_idx = start_idx + params_per_rule
                        if end_idx <= len(param_values):
                            consequent_weights = param_values[start_idx:end_idx]
                except Exception as e:
                    print(f"提取规则{rule_idx}后件参数时出错: {e}")

            rule_data.append({
                'combo': rule_combo,
                'activation': avg_activation,
                'weights': consequent_weights
            })

    # 按激活强度排序
    rule_data.sort(key=lambda x: x['activation'], reverse=True)

    # 输出规则
    print(f"\n{'=' * 80}")
    print(f"前15个最活跃的完整模糊规则:")
    print(f"{'=' * 80}")

    valid_rules_count = 0
    for i, rule_info in enumerate(rule_data[:15]):
        rule_combo = rule_info['combo']
        activation = rule_info['activation']
        weights = rule_info['weights']

        # 构建前件条件
        conditions = []
        for j, mf_idx in enumerate(rule_combo):
            feature_name = input_features[j]
            mf_label = mf_labels[mf_idx]
            conditions.append(f"{feature_name}={mf_label}")

        condition_str = " AND ".join(conditions)

        # 构建后件线性函数
        if weights is not None and len(weights) >= n_inputs + 1:
            valid_rules_count += 1
            if weights.ndim > 1:
                weights = weights.flatten()

            consequent_parts = []
            w0 = weights[0]
            consequent_parts.append(f"{w0:.4f}")

            for j in range(n_inputs):
                if j + 1 < len(weights):
                    w = weights[j + 1]
                    sign = "+" if w >= 0 else ""
                    consequent_parts.append(f"{sign}{w:.4f}×{input_features[j][:6]}")

            consequent_str = " ".join(consequent_parts)
        else:
            consequent_str = "后件参数未找到"

        print(f"\n规则 {i + 1} (激活强度: {activation:.4f}):")
        print(f"IF ({condition_str})")
        print(f"THEN y = {consequent_str}")

    print(f"\n总结: 成功提取 {valid_rules_count}/15 个规则的完整参数")

    return rule_data


def extract_complete_fuzzy_rules_v2(model, input_features):
    """改进版：基于实际参数结构提取TSK模糊规则"""
    print(f"\n{'=' * 80}")
    print(f"SANFIS TSK型模糊规则提取 (改进版)")
    print(f"{'=' * 80}")

    n_inputs = len(input_features)
    n_mf = n_memb_funcs_per_input

    # 生成所有规则组合
    rule_combinations = list(product(range(n_mf), repeat=n_inputs))
    total_rules = len(rule_combinations)

    print(f"输入变量数量: {n_inputs}")
    print(f"每个输入的隶属函数数量: {n_mf}")
    print(f"总规则数量: {total_rules}")

    # 收集所有参数
    all_params = {}
    for name, param in model.named_parameters():
        all_params[name] = param.detach().cpu().numpy()

    print(f"\n找到的所有参数:")
    for name, param in all_params.items():
        print(f"  {name}: shape {param.shape}")

    # 更灵活的参数分类
    mu_params = []
    sigma_params = []
    potential_consequent_params = []

    for name, param in all_params.items():
        param_lower = name.lower()

        # 隶属函数中心参数
        if any(keyword in param_lower for keyword in ['mu', 'mean', 'center', 'centre']):
            mu_params.append((name, param))
        # 隶属函数宽度参数
        elif any(keyword in param_lower for keyword in ['sigma', 'std', 'width', 'scale']):
            sigma_params.append((name, param))
        # 可能的后件参数
        elif any(keyword in param_lower for keyword in ['weight', 'linear', 'consequent', 'coeff', 'output']):
            potential_consequent_params.append((name, param))
        # 如果参数维度看起来像后件参数
        elif param.ndim >= 2 and (param.shape[0] == total_rules or param.shape[1] == n_inputs + 1):
            potential_consequent_params.append((name, param))

    print(f"\n参数分类结果:")
    print(f"  隶属函数中心(mu): {len(mu_params)} 组")
    print(f"  隶属函数宽度(sigma): {len(sigma_params)} 组")
    print(f"  可能的后件参数: {len(potential_consequent_params)} 组")

    for name, param in potential_consequent_params:
        print(f"    {name}: {param.shape}")

    # 尝试从模型的forward方法或特定属性获取后件参数
    consequent_weights = None

    # 方法1: 检查是否有直接的后件参数属性
    if hasattr(model, 'consequent_params'):
        consequent_weights = model.consequent_params.detach().cpu().numpy()
        print(f"  方法1成功: 找到consequent_params {consequent_weights.shape}")
    elif hasattr(model, 'linear_params'):
        consequent_weights = model.linear_params.detach().cpu().numpy()
        print(f"  方法1成功: 找到linear_params {consequent_weights.shape}")
    elif hasattr(model, 'output_weights'):
        consequent_weights = model.output_weights.detach().cpu().numpy()
        print(f"  方法1成功: 找到output_weights {consequent_weights.shape}")

    # 方法2: 从potential_consequent_params中选择最合适的
    if consequent_weights is None and len(potential_consequent_params) > 0:
        for name, param in potential_consequent_params:
            # 检查参数形状是否合理
            if param.ndim == 2:
                if param.shape[0] == total_rules and param.shape[1] >= n_inputs:
                    consequent_weights = param
                    print(f"  方法2成功: 使用 {name} {param.shape}")
                    break
                elif param.shape[1] == total_rules and param.shape[0] >= n_inputs:
                    consequent_weights = param.T  # 转置
                    print(f"  方法2成功: 使用转置的 {name} {param.shape}")
                    break

    # 方法3: 如果还没找到，尝试通过模型推理来估计
    if consequent_weights is None:
        print("  方法3: 尝试通过模型推理估计后件参数...")
        consequent_weights = estimate_consequent_params(model, input_features, rule_combinations)

    # 定义隶属函数标签
    mf_labels = ['低', '中', '高']

    # 计算规则激活强度
    model.eval()
    rule_data = []

    with torch.no_grad():
        sample_size = min(20, len(X_test_tensor))
        X_sample = X_test_tensor[:sample_size]
        S_sample = S_test_tensor[:sample_size]

        for rule_idx, rule_combo in enumerate(rule_combinations):
            # 计算前件激活强度
            total_activation = 0

            for sample_idx in range(len(X_sample)):
                rule_activation = 1.0

                for input_idx, mf_idx in enumerate(rule_combo):
                    x_val = X_sample[sample_idx, input_idx].item()

                    # 使用提取的隶属函数参数
                    if input_idx < len(mu_params) and input_idx < len(sigma_params):
                        _, mu_vals = mu_params[input_idx]
                        _, sigma_vals = sigma_params[input_idx]

                        mu_vals = mu_vals.flatten()
                        sigma_vals = sigma_vals.flatten()

                        if mf_idx < len(mu_vals) and mf_idx < len(sigma_vals):
                            mu = mu_vals[mf_idx]
                            sigma = sigma_vals[mf_idx]
                            membership = np.exp(-0.5 * ((x_val - mu) / (sigma + 1e-8)) ** 2)
                            rule_activation *= membership

                total_activation += rule_activation

            avg_activation = total_activation / len(X_sample)

            # 提取该规则的后件参数
            rule_weights = None
            if consequent_weights is not None:
                try:
                    if consequent_weights.ndim == 2:
                        if rule_idx < consequent_weights.shape[0]:
                            rule_weights = consequent_weights[rule_idx]
                    elif consequent_weights.ndim == 1:
                        # 假设所有规则共享相同的权重结构
                        params_per_rule = n_inputs + 1
                        start_idx = rule_idx * params_per_rule
                        end_idx = start_idx + params_per_rule
                        if end_idx <= len(consequent_weights):
                            rule_weights = consequent_weights[start_idx:end_idx]
                except Exception as e:
                    print(f"提取规则{rule_idx}参数时出错: {e}")

            rule_data.append({
                'combo': rule_combo,
                'activation': avg_activation,
                'weights': rule_weights
            })

    # 按激活强度排序
    rule_data.sort(key=lambda x: x['activation'], reverse=True)

    # 输出规则
    print(f"\n{'=' * 80}")
    print(f"前15个最活跃的完整模糊规则:")
    print(f"{'=' * 80}")

    valid_rules_count = 0
    for i, rule_info in enumerate(rule_data[:15]):
        rule_combo = rule_info['combo']
        activation = rule_info['activation']
        weights = rule_info['weights']

        # 构建前件条件
        conditions = []
        for j, mf_idx in enumerate(rule_combo):
            feature_name = input_features[j]
            mf_label = mf_labels[mf_idx]
            conditions.append(f"{feature_name}={mf_label}")

        condition_str = " AND ".join(conditions)

        # 构建后件线性函数
        if weights is not None and len(weights) >= n_inputs:
            valid_rules_count += 1
            if weights.ndim > 1:
                weights = weights.flatten()

            # 检查是否包含偏置项
            if len(weights) == n_inputs + 1:
                # 包含偏置项
                consequent_parts = [f"{weights[0]:.4f}"]
                for j in range(n_inputs):
                    w = weights[j + 1]
                    sign = "+" if w >= 0 else ""
                    consequent_parts.append(f"{sign}{w:.4f}×{input_features[j][:6]}")
            else:
                # 不包含偏置项
                consequent_parts = []
                for j in range(min(n_inputs, len(weights))):
                    w = weights[j]
                    if j == 0:
                        consequent_parts.append(f"{w:.4f}×{input_features[j][:6]}")
                    else:
                        sign = "+" if w >= 0 else ""
                        consequent_parts.append(f"{sign}{w:.4f}×{input_features[j][:6]}")

            consequent_str = " ".join(consequent_parts)
        else:
            consequent_str = "后件参数未找到或格式不正确"

        print(f"\n规则 {i + 1} (激活强度: {activation:.4f}):")
        print(f"IF ({condition_str})")
        print(f"THEN y = {consequent_str}")

    print(f"\n总结: 成功提取 {valid_rules_count}/15 个规则的完整参数")

    return rule_data


def estimate_consequent_params(model, input_features, rule_combinations):
    """通过数值方法估计后件参数"""
    print("  尝试通过数值方法估计后件参数...")

    n_inputs = len(input_features)
    n_rules = len(rule_combinations)

    # 创建一些测试输入
    test_inputs = []
    for i in range(10):
        test_input = torch.rand(1, n_inputs)
        test_inputs.append(test_input)

    model.eval()

    # 如果能够访问模型的中间层输出，可以尝试估计参数
    # 这里返回None，表示无法估计
    print("  数值估计方法暂未实现")
    return None


# 先调用调试函数
print("=" * 60)
print("开始调试模型参数结构...")
print("=" * 60)




def analyze_consequent_parameters(rule_data, input_features):
    """分析后件参数的统计特性"""
    print(f"\n{'=' * 80}")
    print(f"后件参数统计分析:")
    print(f"{'=' * 80}")

    # 收集所有有效的权重
    all_weights = []  # 保持为列表，不要转换为numpy数组
    bias_terms = []
    feature_weights = {feature: [] for feature in input_features}

    for rule_info in rule_data:
        weights = rule_info['weights']
        if weights is not None and len(weights) >= len(input_features) + 1:
            if weights.ndim > 1:
                weights = weights.flatten()

            bias_terms.append(weights[0])
            all_weights.append(weights)  # 添加到列表而不是数组

            for j, feature in enumerate(input_features):
                if j + 1 < len(weights):
                    feature_weights[feature].append(weights[j + 1])

    # 修复：检查列表长度而不是直接检查数组
    if len(all_weights) > 0:  # 修改这里
        # 只有在需要时才转换为numpy数组
        all_weights_array = np.array(all_weights)

        print(f"有效规则数量: {len(all_weights)}")
        print(f"\n偏置项 (w0) 统计:")
        print(f"  均值: {np.mean(bias_terms):.4f}")
        print(f"  标准差: {np.std(bias_terms):.4f}")
        print(f"  范围: [{np.min(bias_terms):.4f}, {np.max(bias_terms):.4f}]")

        print(f"\n各输入特征权重统计:")
        for feature in input_features:
            weights_list = feature_weights[feature]
            if len(weights_list) > 0:  # 修改这里也用len()检查
                print(f"  {feature}:")
                print(f"    均值: {np.mean(weights_list):.4f}")
                print(f"    标准差: {np.std(weights_list):.4f}")
                print(f"    范围: [{np.min(weights_list):.4f}, {np.max(weights_list):.4f}]")
                print(f"    重要性: {np.mean(np.abs(weights_list)):.4f}")
    else:
        print("没有找到有效的后件参数")
        return

    # 可视化权重分布
    if len(all_weights) > 0:  # 修改这里
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        # 偏置项分布
        if len(bias_terms) > 0:  # 添加检查
            axes[0].hist(bias_terms, bins=min(20, len(bias_terms)), alpha=0.7, edgecolor='black')
            axes[0].set_title('偏置项 (w0) 分布')
            axes[0].set_xlabel('权重值')
            axes[0].set_ylabel('频次')
            axes[0].grid(True, alpha=0.3)

        # 各特征权重分布
        for i, feature in enumerate(input_features):
            ax_idx = i + 1
            if ax_idx < len(axes):
                weights_list = feature_weights[feature]
                if len(weights_list) > 0:  # 修改这里
                    axes[ax_idx].hist(weights_list, bins=min(20, len(weights_list)),
                                      alpha=0.7, edgecolor='black')
                    axes[ax_idx].set_title(f'{feature} 权重分布')
                    axes[ax_idx].set_xlabel('权重值')
                    axes[ax_idx].set_ylabel('频次')
                    axes[ax_idx].grid(True, alpha=0.3)

        # 权重重要性对比
        if len(axes) > len(input_features) + 1:
            importance_scores = []
            feature_names = []
            for feature in input_features:
                weights_list = feature_weights[feature]
                if len(weights_list) > 0:  # 修改这里
                    importance = np.mean(np.abs(weights_list))
                    importance_scores.append(importance)
                    feature_names.append(feature)

            if len(importance_scores) > 0:  # 修改这里
                axes[-1].bar(range(len(feature_names)), importance_scores, alpha=0.7)
                axes[-1].set_xticks(range(len(feature_names)))
                axes[-1].set_xticklabels(feature_names, rotation=45)
                axes[-1].set_title('特征权重重要性 (绝对值均值)')
                axes[-1].set_ylabel('重要性分数')
                axes[-1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        print("没有足够的数据进行可视化")



# [在训练完成后调用这些函数]
print(f"\n开始提取完整的TSK模糊规则...")

# 提取完整规则
# rule_data = extract_complete_fuzzy_rules(model, input_features)
debug_model_parameters(model)
# 然后调用改进的规则提取函数
rule_data = extract_complete_fuzzy_rules_v2(model, input_features)

# 分析后件参数
analyze_consequent_parameters(rule_data, input_features)

