import numpy as np
import pandas as pd
import torch
import torch_optimizer  # 需要安装: pip install torch-optimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sanfis import SANFIS
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import copy
import os
from datetime import datetime
import json
import pickle
from itertools import product

# ===== Matplotlib Configuration =====
import matplotlib as mpl
# 自定义图例处理器类
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

# 确保中文字体和符号正常显示
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica World', 'Arial', 'Arial Unicode MS', 'SimHei']
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['path.simplify'] = True
plt.rcParams['path.snap'] = True

# 创建保存目录
save_dir = 'sanfis_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# [数据加载和预处理代码保持不变...]
# 加载数据
file_path = './data/脱硫数据整理2.xlsx'
try:
    data_df = pd.read_excel(file_path, sheet_name='Sheet1')
except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径：{file_path}")
    exit()

# ==============================================================================
# 1. 新增：引入您提供的参数配置
# ==============================================================================
best_params = {
    "role_GasInletFlow": "both",
    "role_GasInletTemperature": "explanatory",
    "role_GasInletPressure": "state",
    "role_DesulfurizationLiquidFlow": "state",
    "role_DesulfurizationLiquidTemperature": "state",
    "role_DesulfurizationLiquidPressure": "explanatory",
    "role_RotationSpeed": "both",
    "role_HtwoSInletConcentration": "explanatory",
    "M": 3
}

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

# ==============================================================================
# 2. 修改：根据 best_params 动态生成特征列表
# ==============================================================================
# 映射配置键到DataFrame列名 (因为字典键不能有下划线，但列名有)
config_to_df_col_map = {
    'GasInletFlow': 'Gas_Inlet_Flow',
    'GasInletTemperature': 'Gas_Inlet_Temperature',
    'GasInletPressure': 'Gas_Inlet_Pressure',
    'DesulfurizationLiquidFlow': 'Desulfurization_Liquid_Flow',
    'DesulfurizationLiquidTemperature': 'Desulfurization_Liquid_Temperature',
    'DesulfurizationLiquidPressure': 'Desulfurization_Liquid_Pressure',
    'RotationSpeed': 'Rotation_Speed',
    'HtwoSInletConcentration': 'H2S_Inlet_Concentration'
}

s_features = []  # 状态变量 S
x_features = []  # 解释变量 X

print("--- 根据配置解析特征角色 ---")
for key, role in best_params.items():
    if key.startswith("role_"):
        config_key = key.replace("role_", "")
        df_col = config_to_df_col_map.get(config_key)
        if df_col:
            if role == 'state':
                s_features.append(df_col)
            elif role == 'explanatory':
                x_features.append(df_col)
            elif role == 'both':
                s_features.append(df_col)
                x_features.append(df_col)

# 去重并排序，确保列表唯一且顺序固定
s_features = sorted(list(set(s_features)))
x_features = sorted(list(set(x_features)))

# 获取所有需要用到的特征列
all_model_features = sorted(list(set(s_features + x_features)))
output_feature = 'H2S_Outlet_Concentration'

print("\n--- 最终特征列表 ---")
print(f"解释变量 (X): {x_features}")
print(f"状态变量 (S): {s_features}")

# 数据清理和标准化
# ==============================================================================
# 3. 修改：使用新的特征列表进行数据清理和标准化
# ==============================================================================
data_clean = data_df[all_model_features + [output_feature]].dropna()

# 根据新的特征列表准备 S, X, y 数据
S_raw = data_clean[x_features].values
X_raw = data_clean[s_features].values
y_raw = data_clean[output_feature].values.reshape(-1, 1)

# 为 S, X, y 分别创建和拟合缩放器
scaler_S = MinMaxScaler()
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

S_scaled = scaler_S.fit_transform(S_raw)
X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# 同时对 S, X, y进行训练集和测试集划分，确保数据对齐
X_train_np, X_test_np, S_train_np, S_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, S_scaled, y_scaled, test_size=0.2, random_state=42
)

# 转换为PyTorch Tensors
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
S_train_tensor = torch.tensor(S_train_np, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)
S_test_tensor = torch.tensor(S_test_np, dtype=torch.float32)

train_data = [S_train_tensor, X_train_tensor, y_train_tensor]
valid_data = [S_test_tensor, X_test_tensor, y_test_tensor]

# ==============================================================================
# 4. 修改：使用 best_params 中的 M 配置模型
# ==============================================================================
# n_input_features 现在是解释变量X的数量
n_input_features = len(s_features)
# n_memb_funcs_per_input 来自配置
n_memb_funcs_per_input = best_params['M']

print("\n--- 模型配置 ---")
print(f"输入特征数 (解释变量X的数量): {n_input_features}")
print(f"每个输入的隶属函数数量 (M): {n_memb_funcs_per_input}\n")

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

# 定义优化器配置
optimizer_configs = {
    'Adam': {'class': torch.optim.Adam, 'params': {'lr': 0.001}},
    'SGD': {'class': torch.optim.SGD, 'params': {'lr': 0.01, 'momentum': 0.9}},
    'RMSprop': {'class': torch.optim.RMSprop, 'params': {'lr': 0.001}},
    'AdaBelief': {'class': torch_optimizer.AdaBelief, 'params': {'lr': 0.001}},
    'AccSGD': {'class': torch_optimizer.AccSGD, 'params': {'lr': 0.01}},
    'AdaBound': {'class': torch_optimizer.AdaBound, 'params': {'lr': 0.001}},
    'RAdam': {'class': torch_optimizer.RAdam, 'params': {'lr': 0.001}},
}


# 完整的TSK模糊规则提取函数
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


import numpy as np
from itertools import product
import torch


# 假设 n_memb_funcs_per_input 是一个在函数外部定义的全局变量或可访问变量
# 例如: n_memb_funcs_per_input = 3


import numpy as np
from itertools import product
import torch

import numpy as np
from itertools import product
import torch


def extract_complete_fuzzy_rules_v2(model, premise_features, consequent_features):
    """
    最终精确适配版 TSK 模糊规则提取函数。
    - 完全根据您提供的日志中的参数名和形状进行设计。
    - 正确处理转置的后件权重矩阵。
    """
    print(f"\n{'=' * 80}")
    print(f"SANFIS TSK型模糊规则提取 (精确适配版)")
    print(f"{'=' * 80}")

    n_premise_inputs = len(premise_features)
    n_consequent_inputs = len(consequent_features)
    try:
        n_mf = best_params['M']
    except NameError:
        n_mf = 3
        print(f"警告: 无法访问全局 best_params['M']，使用默认值 n_mf={n_mf}")

    rule_combinations = list(product(range(n_mf), repeat=n_premise_inputs))
    total_rules = len(rule_combinations)

    print(f"前件变量 (S): {n_premise_inputs} -> {premise_features}")
    print(f"后件变量 (X): {n_consequent_inputs} -> {consequent_features}")
    print(f"每个前件变量的隶属函数数量 (M): {n_mf}")
    print(f"总规则数量 (M ^ |S|): {total_rules}")

    # 1. 提取所有模型参数
    all_params = {name: param.detach().cpu().numpy() for name, param in model.named_parameters()}

    # 2. 根据日志精确查找参数
    mu_params = [all_params.get(f'layers.fuzzylayer.fuzzyfication.{i}._mu', np.zeros(n_mf)).flatten() for i in
                 range(n_premise_inputs)]
    sigma_params = [all_params.get(f'layers.fuzzylayer.fuzzyfication.{i}._sigma', np.ones(n_mf)).flatten() for i in
                    range(n_premise_inputs)]

    # 精确查找后件参数并处理转置
    consequent_weights_raw = all_params.get('layers.consequence._weight')
    consequent_bias_raw = all_params.get('layers.consequence._bias')

    consequent_weights = None
    consequent_bias = None

    if consequent_weights_raw is not None and consequent_bias_raw is not None:
        print(f"信息: 找到后件权重 'layers.consequence._weight' (形状: {consequent_weights_raw.shape})")
        print(f"信息: 找到后件偏置 'layers.consequence._bias' (形状: {consequent_bias_raw.shape})")

        # 核心修正：权重矩阵需要转置！
        # 原始形状: [n_consequent_inputs, total_rules] -> [5, 243]
        # 需要形状: [total_rules, n_consequent_inputs] -> [243, 5]
        if consequent_weights_raw.shape == (n_consequent_inputs, total_rules):
            consequent_weights = consequent_weights_raw.T
            print(f"信息: 已将后件权重转置为 {consequent_weights.shape}")
        else:
            print(
                f"警告: 后件权重形状 {consequent_weights_raw.shape} 与预期 {(n_consequent_inputs, total_rules)} 不符。")
            consequent_weights = np.zeros((total_rules, n_consequent_inputs))

        consequent_bias = consequent_bias_raw.flatten()  # 形状从 [1, 243] 变为 [243,]

    else:
        print("错误: 未能找到 'layers.consequence._weight' 或 'layers.consequence._bias'。")
        consequent_weights = np.zeros((total_rules, n_consequent_inputs))
        consequent_bias = np.zeros(total_rules)

    # 3. 构建规则
    mf_labels = ['低', '中', '高']
    if n_mf > 3: mf_labels += [f'水平{i + 1}' for i in range(3, n_mf)]

    rule_data = []
    for rule_idx, rule_combo in enumerate(rule_combinations):
        conditions = []
        membership_params = {}
        for j, mf_idx in enumerate(rule_combo):
            feature_name = premise_features[j]
            mf_label = mf_labels[mf_idx % len(mf_labels)]
            conditions.append(f"{feature_name}={mf_label}")
            if mf_idx < len(mu_params[j]):
                membership_params[feature_name] = {'mu': float(mu_params[j][mf_idx]),
                                                   'sigma': float(sigma_params[j][mf_idx]), 'label': mf_label}
        condition_str = " AND ".join(conditions)

        consequent_str = "y = "
        consequent_params = {}

        bias = consequent_bias[rule_idx]
        rule_w = consequent_weights[rule_idx]

        consequent_params['bias'] = float(bias)
        consequent_str += f"{bias:.4f}"

        for j in range(n_consequent_inputs):
            weight = rule_w[j]
            feature_name = consequent_features[j]
            consequent_params[f'weight_{feature_name}'] = float(weight)
            sign = " + " if weight >= 0 else " - "
            consequent_str += f"{sign}{abs(weight):.4f}×{feature_name}"

        rule_data.append({
            'rule_id': rule_idx + 1, 'activation': 0.0, 'condition': condition_str,
            'membership_params': membership_params, 'consequent_params': consequent_params,
            'consequent_equation': consequent_str, 'weights': np.insert(rule_w, 0, bias).tolist()
        })

    # 4. 输出规则
    print(f"\n{'=' * 80}\n提取的完整TSK模糊规则 (显示前15条):\n{'=' * 80}")
    for rule_info in rule_data[:15]:
        print(f"\n规则 {rule_info['rule_id']}:")
        print(f"  前件: IF ({rule_info['condition']})")
        print(f"  后件: THEN {rule_info['consequent_equation']}")
    if len(rule_data) > 15: print(f"\n... (共 {len(rule_data)} 条规则)")

    return rule_data


def estimate_consequent_params(model, input_features, rule_combinations):
    """通过数值方法估计后件参数"""
    print("  尝试通过数值方法估计后件参数...")
    return None


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


def train_model(optimizer_name, epochs=1000):
    """训练单个模型"""
    print(f"\n{'=' * 60}")
    print(f"训练配置: 优化器={optimizer_name}")
    print(f"{'=' * 60}")

    # 初始化模型
    model = SANFIS(
        membfuncs=membfuncs_config,
        n_input=n_input_features,
        to_device='cpu',
        scale='Std'
    )

    # 保存训练前的参数
    initial_params = extract_membership_params_alternative(model, membfuncs_config)

    # 训练历史记录
    training_history = {
        'epochs': [],
        'mu_params': [],
        'sigma_params': [],
        'train_losses': [],
        'valid_losses': []
    }

    # 设置优化器
    optimizer_config = optimizer_configs[optimizer_name]
    try:
        optimizer = optimizer_config['class'](model.parameters(), **optimizer_config['params'])
    except Exception as e:
        print(f"创建优化器 {optimizer_name} 时出错: {e}")
        return None

    loss_function = torch.nn.MSELoss(reduction='mean')

    print_interval = 100
    save_interval = 200

    model.train()
    for epoch in range(epochs):
        # 训练步骤
        optimizer.zero_grad()
        train_pred = model(train_data[0], train_data[1])
        train_loss = loss_function(train_pred, train_data[2])
        train_loss.backward()

        # 梯度裁剪
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

    # 最终评估
    model.eval()
    with torch.no_grad():
        y_pred_scaled_tensor = model(S_test_tensor, X_test_tensor)

    y_pred_scaled = y_pred_scaled_tensor.detach().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_np)

    mse = mean_squared_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    final_params = extract_membership_params_alternative(model, membfuncs_config)

    # 提取完整的TSK模糊规则
    print(f"\n开始提取完整的TSK模糊规则...")
    debug_model_parameters(model)
    fuzzy_rules = extract_complete_fuzzy_rules_v2(model, s_features, x_features)

    result = {
        'optimizer_name': optimizer_name,
        'mse': mse,
        'r2': r2,
        'initial_params': initial_params,
        'final_params': final_params,
        'training_history': training_history,
        'y_pred': y_pred,
        'y_test_original': y_test_original,
        'fuzzy_rules': fuzzy_rules,
        'model_state_dict': model.state_dict(),
        'trained_model': model  # 保存训练好的模型
    }

    print(f"最终结果: MSE={mse:.4f}, R²={r2:.4f}")

    return result


class HandlerDashedSolidLine(HandlerBase):
    """
    自定义图例处理器，用于创建上虚下实的双线图例符号。
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # 上方的虚线 (代表训练前)
        line1 = Line2D([0, width], [height * 0.66, height * 0.66], linestyle='--', color=orig_handle.get_color())
        # 下方的实线 (代表训练后)
        line2 = Line2D([0, width], [height * 0.33, height * 0.33], linestyle='-', color=orig_handle.get_color())
        return [line1, line2]


# ==============================================================================
#  ↓↓↓ 主要修改区域：合并绘图函数 ↓↓↓
# ==============================================================================

def plot_combined_figure(best_result, rmsprop_result, save_dir, timestamp):
    """
    将RMSprop的损失曲线和最佳模型的隶属度函数演化图合并到一个2x3的图中。
    """
    if rmsprop_result is None:
        print("错误：未提供RMSprop的结果，无法生成组合图。")
        return

    print(f"\n正在生成组合图：RMSprop损失曲线 + {best_result['optimizer_name']}隶属度函数...")

    # 1. 创建 2x3 的子图网格
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 7 * n_rows))
    axes = axes.ravel()  # 将2D数组展平为1D，方便索引

    # --- Part A: 在第一个子图 (axes[0]) 中绘制 RMSprop 损失曲线 ---
    ax_loss = axes[0]
    history = rmsprop_result['training_history']
    epochs = history['epochs']
    train_losses = history['train_losses']
    valid_losses = history['valid_losses']

    ax_loss.plot(epochs, train_losses, label='Train Loss', color='#D13C29', linewidth=3, alpha=0.8)
    ax_loss.plot(epochs, valid_losses, label='Validation Loss', color='#5381B2', linewidth=3, alpha=0.8)

    # 统一风格
    ax_loss.set_xlabel('Epochs', fontsize=22)
    ax_loss.set_ylabel('Loss', fontsize=22)
    ax_loss.set_yscale('log')
    ax_loss.legend(fontsize=18)
    ax_loss.grid(False)
    ax_loss.tick_params(axis='both', which='major', direction='out', length=6, color='black', labelsize=18)
    for spine in ax_loss.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    # --- Part B: 在其余子图中绘制最佳模型的隶属度函数 ---
    initial_params = best_result['initial_params']
    final_params = best_result['final_params']
    optimizer_name = best_result['optimizer_name']

    n_inputs = len(x_features)
    x_range = np.linspace(0, 1, 100)

    # 样式定义
    letters = ['$Flow_{\mathrm{liquid}}\ (\mathrm{Standardization})$',
                '$Rotation\ Speed\ (\mathrm{Standardization})$',
               '$Flow_{\mathrm{gas}}\ (\mathrm{Standardization})$',
               '$Temperature_{\mathrm{gas}}\ (\mathrm{Standardization})$',
               '$Pressure_{\mathrm{gas}}\ (\mathrm{Standardization})$'
               ]
    y_label = 'Membership Degree'
    colors = ['royalblue', 'darkorange', 'seagreen']
    mf_names = ['Low', 'Medium', 'High']

    for i in range(n_inputs):
        ax_memb = axes[i + 1]  # 从第二个子图开始绘制

        mu_init = initial_params['mu_params'][i]
        sigma_init = initial_params['sigma_params'][i]
        mu_final = final_params['mu_params'][i]
        sigma_final = final_params['sigma_params'][i]
        n_funcs = min(len(mu_init), len(sigma_init))

        legend_handles = []
        legend_labels = []

        for j in range(n_funcs):
            color = colors[j % len(colors)]
            # 训练前 (虚线)
            membership_init = np.exp(-0.5 * ((x_range - mu_init[j]) / (sigma_init[j] + 1e-8)) ** 2)
            ax_memb.plot(x_range, membership_init, '--', color=color, linewidth=2.5, alpha=1)
            # 训练后 (实线)
            membership_final = np.exp(-0.5 * ((x_range - mu_final[j]) / (sigma_final[j] + 1e-8)) ** 2)
            ax_memb.plot(x_range, membership_final, '-', color=color, linewidth=3, alpha=1)

            # 创建自定义图例
            proxy_artist = plt.Line2D([0], [0], color=color)
            legend_handles.append(proxy_artist)
            init_formula = rf"e^{{-\frac{{(x - {mu_init[j]:.2f})^2}}{{2 \cdot {sigma_init[j]:.2f}^2}}}}"
            final_formula = rf"e^{{-\frac{{(x - {mu_final[j]:.2f})^2}}{{2 \cdot {sigma_final[j]:.2f}^2}}}}"
            label_text = f"${init_formula}$" + r" $\rightarrow$ " + f"${final_formula}$"
            legend_labels.append(label_text)

        # 应用统一风格
        ax_memb.set_xlabel(rf'{letters[i]}', fontsize=22)
        ax_memb.set_ylabel(y_label, fontsize=22)
        ax_memb.grid(False)
        ax_memb.set_ylim(-0.1, 1.3)
        ax_memb.tick_params(axis='both', which='major', direction='out', length=6, color='black', labelsize=18)
        for spine in ax_memb.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        ax_memb.legend(handles=legend_handles, labels=legend_labels, handler_map={plt.Line2D: HandlerDashedSolidLine()},
                       fontsize=14, ncol=len(legend_labels), loc='upper center', columnspacing=1.4, handletextpad=0.6)

    # 隐藏任何未使用的子图
    for i in range(1 + n_inputs, n_rows * n_cols):
        axes[i].set_visible(False)

    # 保存并显示最终的组合图
    plt.tight_layout(pad=3.0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 使用一个能反映内容的文件名
    filename = f'combined_loss_membership_{optimizer_name}_{timestamp}'
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{filename}.pdf'), bbox_inches='tight')
    print(f"组合图已保存为: {os.path.join(save_dir, f'{filename}.png')}")
    plt.show()


def plot_parameter_trends(best_result, save_dir, timestamp):
    """绘制最佳优化器的参数和损失变化趋势图。"""
    training_history = best_result['training_history']
    optimizer_name = best_result['optimizer_name']

    if len(training_history['epochs']) <= 1:
        print("训练历史数据不足，跳过参数趋势图绘制。")
        return

    print(f"绘制 {optimizer_name} 的参数变化趋势...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    epochs = training_history['epochs']
    initial_params = best_result['initial_params']
    n_inputs = len(x_features)

    # μ参数变化
    for i in range(n_inputs):
        mu_evolution = [np.mean(epoch_params[i]) for epoch_params in training_history['mu_params']]
        ax1.plot(epochs, mu_evolution, 'o-', label=f'{x_features[i]}', linewidth=2)
    ax1.set_title(f'{optimizer_name} μ参数变化趋势', fontweight='bold')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('平均μ值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # σ参数变化
    for i in range(n_inputs):
        sigma_evolution = [np.mean(epoch_params[i]) for epoch_params in training_history['sigma_params']]
        ax2.plot(epochs, sigma_evolution, 's-', label=f'{x_features[i]}', linewidth=2)
    ax2.set_title(f'{optimizer_name} σ参数变化趋势', fontweight='bold')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('平均σ值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 损失变化
    ax3.plot(epochs, training_history['train_losses'], 'b-', label='训练损失', linewidth=2)
    ax3.plot(epochs, training_history['valid_losses'], 'r-', label='验证损失', linewidth=2)
    ax3.set_title(f'{optimizer_name} 损失函数变化', fontweight='bold')
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
            total_change = sum(
                abs(np.mean(training_history['mu_params'][i][j]) - np.mean(initial_params['mu_params'][j])) for j in
                range(n_inputs))
            param_changes.append(total_change / n_inputs)
    ax4.plot(epochs, param_changes, 'g-', linewidth=2, marker='o')
    ax4.set_title(f'{optimizer_name} 参数变化幅度', fontweight='bold')
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('相对变化幅度')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'parameter_trends_{optimizer_name}_{timestamp}.png'), dpi=300,
                bbox_inches='tight')
    plt.show()


# ==============================================================================
#  ↑↑↑ 主要修改区域：合并绘图函数 ↑↑↑
# ==============================================================================

def save_complete_fuzzy_rules(results_list, save_dir):
    """保存完整的TSK模糊规则到文件"""
    rules_dir = os.path.join(save_dir, 'complete_fuzzy_rules')
    if not os.path.exists(rules_dir):
        os.makedirs(rules_dir)

    for result in results_list:
        optimizer_name = result['optimizer_name']
        fuzzy_rules = result['fuzzy_rules']

        # 保存为JSON格式（完整数据）
        rules_file = os.path.join(rules_dir, f'complete_rules_{optimizer_name}_{timestamp}.json')
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(fuzzy_rules, f, ensure_ascii=False, indent=2)

        # 保存为详细的可读文本格式
        txt_file = os.path.join(rules_dir, f'complete_rules_{optimizer_name}_{timestamp}.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"SANFIS完整TSK模糊规则 - 优化器: {optimizer_name}\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"模型性能: MSE={result['mse']:.6f}, R²={result['r2']:.6f}\n")
            f.write("=" * 80 + "\n\n")

            # 按激活强度排序显示前20条规则
            sorted_rules = sorted(fuzzy_rules, key=lambda x: x['activation'], reverse=True)

            for i, rule in enumerate(sorted_rules[:20]):
                f.write(f"规则 {rule['rule_id']} (激活强度: {rule['activation']:.6f}):\n")
                f.write(f"前件: IF ({rule['condition']})\n")
                f.write(f"后件: THEN y = {rule['consequent_equation']}\n\n")

                f.write("详细参数:\n")
                f.write("  隶属函数参数:\n")
                for feature, params in rule['membership_params'].items():
                    f.write(f"    {feature}: μ={params['mu']:.6f}, σ={params['sigma']:.6f}, 标签={params['label']}\n")

                f.write("  后件参数:\n")
                for param_name, value in rule['consequent_params'].items():
                    f.write(f"    {param_name}: {value:.6f}\n")

                f.write("\n" + "-" * 60 + "\n\n")

        print(f"完整TSK模糊规则已保存: {rules_file}")
        print(f"可读格式已保存: {txt_file}")


def save_results(results, experiment_name, save_dir, timestamp):
    """保存实验结果"""
    exp_dir = os.path.join(save_dir, f'{experiment_name}_{timestamp}')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # 保存结果数据（不包含模型对象，避免序列化问题）
    results_to_save = []
    for result in results:
        result_copy = result.copy()
        if 'trained_model' in result_copy:
            del result_copy['trained_model']  # 移除模型对象
        results_to_save.append(result_copy)

    with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_to_save, f)

    # 保存配置信息
    config = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'input_features': x_features,
        'n_input_features': n_input_features,
        'n_memb_funcs_per_input': n_memb_funcs_per_input,
        'membfuncs_config': membfuncs_config
    }

    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return exp_dir


def generate_summary_report(results_list, save_dir):
    """生成总结报告"""
    report_file = os.path.join(save_dir, f'summary_report_{timestamp}.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SANFIS优化器比较实验总结报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"实验时间: {timestamp}\n")
        f.write(f"数据集: 脱硫数据\n")
        f.write(f"输入特征: {', '.join(x_features)}\n")
        f.write(f"输出特征: {output_feature}\n\n")

        f.write("实验结果:\n")
        f.write("-" * 40 + "\n")

        # 按R²排序
        sorted_results = sorted(results_list, key=lambda x: x['r2'], reverse=True)

        for i, result in enumerate(sorted_results):
            f.write(f"{i + 1}. {result['optimizer_name']}\n")
            f.write(f"   MSE: {result['mse']:.6f}\n")
            f.write(f"   R²:  {result['r2']:.6f}\n")
            f.write(f"   TSK规则数: {len(result['fuzzy_rules'])}\n\n")

        f.write(f"最佳优化器: {sorted_results[0]['optimizer_name']} ")
        f.write(f"(R² = {sorted_results[0]['r2']:.6f})\n")

        # 性能改进分析
        best_r2 = sorted_results[0]['r2']
        worst_r2 = sorted_results[-1]['r2']
        improvement = ((best_r2 - worst_r2) / abs(worst_r2)) * 100 if worst_r2 != 0 else float('inf')
        f.write(f"性能改进: {improvement:.2f}%\n")

    print(f"总结报告已保存: {report_file}")


# 主实验流程
def run_optimizer_comparison():
    """运行优化器比较实验"""
    print(f"\n{'=' * 80}")
    print(f"开始优化器比较实验")
    print(f"{'=' * 80}")

    results_list = []

    # 选择要比较的优化器
    selected_optimizers = ['Adam', 'SGD', 'RMSprop', 'AdaBelief', 'AccSGD', 'RAdam']

    for optimizer_name in selected_optimizers:
        try:
            print(f"\n正在训练: {optimizer_name}")
            result = train_model(optimizer_name, epochs=3500)
            if result is not None:
                results_list.append(result)
        except Exception as e:
            print(f"训练 {optimizer_name} 时出错: {e}")
            continue

    if not results_list:
        print("没有成功训练的模型！")
        return

    save_dir = 'sanfis_process_optimization'
    # 找到最佳结果
    best_result = max(results_list, key=lambda x: x['r2'])
    # 专门找到RMSprop的结果
    rmsprop_result = next((r for r in results_list if r['optimizer_name'].lower() == 'rmsprop'), None)

    print(f"\n最佳优化器: {best_result['optimizer_name']} (R² = {best_result['r2']:.6f})")

    # 创建实验目录
    exp_dir = save_results(results_list, 'optimizer_comparison', save_dir, timestamp)

    # 保存最佳模型
    model_save_path = os.path.join(exp_dir, f'best_model_{best_result["optimizer_name"]}.pth')
    torch.save({
        'model_state_dict': best_result['trained_model'].state_dict(),
        'scaler_S': scaler_S,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        's_features': s_features,
        'x_features': x_features,
        'output_feature': output_feature,
        'best_params_config': best_params
    }, model_save_path)
    print(f"\n训练好的最佳模型已保存: {model_save_path}")

    # ==============================================================================
    #  ↓↓↓ 主要修改区域：调用新的绘图函数 ↓↓↓
    # ==============================================================================
    # 生成组合图表
    print(f"\n生成图表...")
    plot_combined_figure(best_result, rmsprop_result, exp_dir, timestamp)

    # 绘制最佳优化器的参数变化趋势
    plot_parameter_trends(best_result, exp_dir, timestamp)

    # 保存完整的TSK模糊规则
    print(f"\n保存完整的TSK模糊规则...")
    save_complete_fuzzy_rules(results_list, exp_dir)

    # 生成总结报告
    generate_summary_report(results_list, exp_dir)

    print(f"\n实验完成！结果保存在: {exp_dir}")

    return results_list, exp_dir


def run_single_experiment():
    """
    运行单次模型训练实验，并生成所有结果和图表。
    """
    print(f"\n{'=' * 80}")
    print(f"开始单次模型训练实验")
    print(f"{'=' * 80}")

    # ==============================================================================
    #  1. 指定要使用的优化器和训练轮数
    # ==============================================================================
    optimizer_to_use = 'Adam'  # 您可以换成 'RMSprop' 或其他任何一个
    training_epochs = 3500

    result = None
    try:
        print(f"\n正在使用优化器: {optimizer_to_use}")
        result = train_model(optimizer_to_use, epochs=training_epochs)
    except Exception as e:
        print(f"训练过程中发生严重错误: {e}")
        # 引入 traceback 模块可以打印更详细的错误堆栈
        import traceback
        traceback.print_exc()

    # 检查训练是否成功
    if result is None:
        print("\n实验失败：模型未能成功训练。请检查上面的错误信息。")
        return None, None

    print("\n模型训练成功！")
    print(f"最终性能: MSE={result['mse']:.6f}, R²={result['r2']:.6f}")

    # ==============================================================================
    #  2. 设置保存目录和文件名
    # ==============================================================================
    save_dir = 'sanfis_single_run_results'
    experiment_name = f'experiment_{optimizer_to_use}'
    exp_dir = save_results([result], experiment_name, save_dir, timestamp) # save_results 仍接收列表

    # ==============================================================================
    #  3. 保存训练好的模型
    # ==============================================================================
    model_save_path = os.path.join(exp_dir, f'final_model_{optimizer_to_use}.pth')
    torch.save({
        'model_state_dict': result['trained_model'].state_dict(),
        'scaler_S': scaler_S,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        's_features': s_features,
        'x_features': x_features,
        'output_feature': output_feature,
        'best_params_config': best_params
    }, model_save_path)
    print(f"\n训练好的模型已保存: {model_save_path}")

    # ==============================================================================
    #  4. 生成图表和报告
    # ==============================================================================
    print(f"\n生成图表和报告...")

    # plot_combined_figure 需要两个参数，我们可以传入同一个 result
    plot_combined_figure(result, result, exp_dir, timestamp)

    # 绘制参数变化趋势
    plot_parameter_trends(result, exp_dir, timestamp)

    # 保存完整的TSK模糊规则
    save_complete_fuzzy_rules([result], exp_dir) # save_complete_fuzzy_rules 接收列表

    # 生成总结报告
    generate_summary_report([result], exp_dir) # generate_summary_report 接收列表

    print(f"\n实验完成！所有结果保存在: {exp_dir}")

    return result, exp_dir

# 运行实验
# if __name__ == "__main__":
#     results_list, exp_dir = run_optimizer_comparison()
#
#     # 显示最佳结果
#     if results_list:
#         best_result = max(results_list, key=lambda x: x['r2'])
#         print(f"\n{'=' * 60}")
#         print(f"最佳结果:")
#         print(f"优化器: {best_result['optimizer_name']}")
#         print(f"MSE: {best_result['mse']:.6f}")
#         print(f"R²: {best_result['r2']:.6f}")
#         print(f"TSK规则数量: {len(best_result['fuzzy_rules'])}")
#         print(f"{'=' * 60}")




# ==============================================================================
#  主程序入口
# ==============================================================================
if __name__ == "__main__":
    # 运行单个实验
    final_result, experiment_directory = run_single_experiment()

    # 显示最终结果
    if final_result:
        print(f"\n{'=' * 60}")
        print(f"最终实验结果:")
        print(f"  优化器: {final_result['optimizer_name']}")
        print(f"  MSE: {final_result['mse']:.6f}")
        print(f"  R²: {final_result['r2']:.6f}")
        print(f"  TSK规则数量: {len(final_result['fuzzy_rules'])}")
        print(f"  结果目录: {experiment_directory}")
        print(f"{'=' * 60}")

