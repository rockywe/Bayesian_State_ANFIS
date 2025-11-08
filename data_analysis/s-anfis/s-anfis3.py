import numpy as np
import pandas as pd
import torch
import torch_optimizer  # 需要安装: pip install torch-optimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sanfis import SANFIS, plottingtools
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import copy
import os
from datetime import datetime
import json
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

# 定义优化器和激活函数配置
optimizer_configs = {
    'Adam': {'class': torch.optim.Adam, 'params': {'lr': 0.001}},
    'SGD': {'class': torch.optim.SGD, 'params': {'lr': 0.01, 'momentum': 0.9}},
    'RMSprop': {'class': torch.optim.RMSprop, 'params': {'lr': 0.001}},
    'AdaBelief': {'class': torch_optimizer.AdaBelief, 'params': {'lr': 0.001}},
    'AccSGD': {'class': torch_optimizer.AccSGD, 'params': {'lr': 0.01}},
    'AdaBound': {'class': torch_optimizer.AdaBound, 'params': {'lr': 0.001}},
    'RAdam': {'class': torch_optimizer.RAdam, 'params': {'lr': 0.001}},
}

activation_functions = {
    'ReLU': torch.nn.ReLU(),
    'Tanh': torch.nn.Tanh(),
    'Sigmoid': torch.nn.Sigmoid(),
    'LeakyReLU': torch.nn.LeakyReLU(0.1),
    'ELU': torch.nn.ELU(),
    'GELU': torch.nn.GELU(),
}


# 保存函数
def save_results(results, experiment_name, save_dir, timestamp):
    """保存实验结果"""
    exp_dir = os.path.join(save_dir, f'{experiment_name}_{timestamp}')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # 保存结果数据
    with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # 保存配置信息
    config = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'input_features': input_features,
        'n_input_features': n_input_features,
        'n_memb_funcs_per_input': n_memb_funcs_per_input,
        'membfuncs_config': membfuncs_config
    }

    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return exp_dir


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


def train_model(optimizer_name, activation_name=None, epochs=1000):
    """训练单个模型"""
    print(f"\n{'=' * 60}")
    print(f"训练配置: 优化器={optimizer_name}, 激活函数={activation_name}")
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

    # 提取模糊规则
    fuzzy_rules = extract_fuzzy_rules_simple(model, input_features, final_params)

    result = {
        'optimizer_name': optimizer_name,
        'activation_name': activation_name,
        'mse': mse,
        'r2': r2,
        'initial_params': initial_params,
        'final_params': final_params,
        'training_history': training_history,
        'y_pred': y_pred,
        'y_test_original': y_test_original,
        'fuzzy_rules': fuzzy_rules,
        'model_state_dict': model.state_dict()
    }

    print(f"最终结果: MSE={mse:.4f}, R²={r2:.4f}")

    return result


def extract_fuzzy_rules_simple(model, input_features, params):
    """简化的模糊规则提取"""
    from itertools import product

    n_inputs = len(input_features)
    n_mf = n_memb_funcs_per_input
    mf_labels = ['低', '中', '高']

    # 生成所有规则组合
    rule_combinations = list(product(range(n_mf), repeat=n_inputs))

    rules = []
    for rule_idx, rule_combo in enumerate(rule_combinations):
        # 构建前件条件
        conditions = []
        for j, mf_idx in enumerate(rule_combo):
            feature_name = input_features[j]
            mf_label = mf_labels[mf_idx]
            conditions.append(f"{feature_name}={mf_label}")

        condition_str = " AND ".join(conditions)

        # 获取隶属函数参数
        rule_params = {}
        for j, mf_idx in enumerate(rule_combo):
            if j < len(params['mu_params']) and j < len(params['sigma_params']):
                mu_vals = params['mu_params'][j]
                sigma_vals = params['sigma_params'][j]
                if mf_idx < len(mu_vals) and mf_idx < len(sigma_vals):
                    rule_params[f'input_{j}'] = {
                        'mu': float(mu_vals[mf_idx]),
                        'sigma': float(sigma_vals[mf_idx])
                    }

        rules.append({
            'rule_id': rule_idx,
            'condition': condition_str,
            'params': rule_params
        })

    return rules


def plot_comparison_results(results_list, save_dir):
    """绘制比较结果"""
    if not results_list:
        return

    # 1. 性能比较图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # MSE比较
    optimizers = [r['optimizer_name'] for r in results_list]
    mse_values = [r['mse'] for r in results_list]
    r2_values = [r['r2'] for r in results_list]

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    bars1 = ax1.bar(optimizers, mse_values, color=colors, alpha=0.7)
    ax1.set_title('不同优化器的MSE比较', fontweight='bold', fontsize=14)
    ax1.set_ylabel('均方误差 (MSE)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 在柱状图上添加数值标签
    for bar, mse in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mse_values) * 0.01,
                 f'{mse:.4f}', ha='center', va='bottom', fontsize=10)

    # R²比较
    bars2 = ax2.bar(optimizers, r2_values, color=colors, alpha=0.7)
    ax2.set_title('不同优化器的R²比较', fontweight='bold', fontsize=14)
    ax2.set_ylabel('决定系数 (R²)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    for bar, r2 in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(r2_values) * 0.01,
                 f'{r2:.4f}', ha='center', va='bottom', fontsize=10)

    # 训练损失对比
    for i, result in enumerate(results_list):
        history = result['training_history']
        ax3.plot(history['epochs'], history['train_losses'],
                 label=f"{result['optimizer_name']}",
                 color=colors[i], linewidth=2)

    ax3.set_title('训练损失对比', fontweight='bold', fontsize=14)
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('训练损失')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 验证损失对比
    for i, result in enumerate(results_list):
        history = result['training_history']
        ax4.plot(history['epochs'], history['valid_losses'],
                 label=f"{result['optimizer_name']}",
                 color=colors[i], linewidth=2)

    ax4.set_title('验证损失对比', fontweight='bold', fontsize=14)
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('验证损失')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'optimizer_comparison_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 预测结果对比
    n_optimizers = len(results_list)
    n_cols = min(3, n_optimizers)
    n_rows = (n_optimizers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_optimizers == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results_list):
        row = i // n_cols
        col = i % n_cols

        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        y_test = result['y_test_original']
        y_pred = result['y_pred']

        ax.scatter(y_test, y_pred, alpha=0.6, color=colors[i])
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('实际值')
        ax.set_ylabel('预测值')
        ax.set_title(f'{result["optimizer_name"]}\n(R² = {result["r2"]:.4f})')
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_optimizers, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'prediction_comparison_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_membership_evolution_comparison(results_list, save_dir):
    """绘制不同优化器的隶属函数演化对比"""
    if not results_list:
        return

    n_inputs = len(input_features)
    n_optimizers = len(results_list)

    fig, axes = plt.subplots(n_inputs, n_optimizers, figsize=(5 * n_optimizers, 4 * n_inputs))
    if n_inputs == 1:
        axes = axes.reshape(1, -1)
    if n_optimizers == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_optimizers))
    x = np.linspace(0, 1, 100)

    for opt_idx, result in enumerate(results_list):
        initial_params = result['initial_params']
        final_params = result['final_params']
        optimizer_name = result['optimizer_name']

        for input_idx in range(n_inputs):
            ax = axes[input_idx, opt_idx]

            # 初始和最终隶属函数
            if (input_idx < len(initial_params['mu_params']) and
                    input_idx < len(final_params['mu_params'])):

                mu_init = initial_params['mu_params'][input_idx]
                sigma_init = initial_params['sigma_params'][input_idx]
                mu_final = final_params['mu_params'][input_idx]
                sigma_final = final_params['sigma_params'][input_idx]

                n_funcs = min(len(mu_init), len(sigma_init), len(mu_final), len(sigma_final))

                for j in range(n_funcs):
                    # 初始隶属函数
                    membership_init = np.exp(-0.5 * ((x - mu_init[j]) / (sigma_init[j] + 1e-8)) ** 2)
                    ax.plot(x, membership_init, '--', alpha=0.5, linewidth=1,
                            label=f'初始MF{j + 1}' if opt_idx == 0 else "")

                    # 最终隶属函数
                    membership_final = np.exp(-0.5 * ((x - mu_final[j]) / (sigma_final[j] + 1e-8)) ** 2)
                    ax.plot(x, membership_final, '-', linewidth=2,
                            label=f'最终MF{j + 1}' if opt_idx == 0 else "")

            ax.set_title(f'{optimizer_name}\n{input_features[input_idx]}', fontsize=10)
            ax.set_xlabel('标准化输入值')
            ax.set_ylabel('隶属度')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)

            if input_idx == 0 and opt_idx == 0:
                ax.legend(fontsize=8)

    plt.suptitle('不同优化器的隶属函数演化对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'membership_evolution_comparison_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def save_fuzzy_rules(results_list, save_dir):
    """保存模糊规则到文件"""
    rules_dir = os.path.join(save_dir, 'fuzzy_rules')
    if not os.path.exists(rules_dir):
        os.makedirs(rules_dir)

    for result in results_list:
        optimizer_name = result['optimizer_name']
        fuzzy_rules = result['fuzzy_rules']

        # 保存为JSON格式
        rules_file = os.path.join(rules_dir, f'rules_{optimizer_name}_{timestamp}.json')
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(fuzzy_rules, f, ensure_ascii=False, indent=2)

        # 保存为可读文本格式
        txt_file = os.path.join(rules_dir, f'rules_{optimizer_name}_{timestamp}.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"SANFIS模糊规则 - 优化器: {optimizer_name}\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write("=" * 60 + "\n\n")

            for i, rule in enumerate(fuzzy_rules[:20]):  # 只保存前20条规则
                f.write(f"规则 {i + 1}:\n")
                f.write(f"IF ({rule['condition']})\n")
                f.write(f"参数: {rule['params']}\n\n")

        print(f"模糊规则已保存: {rules_file}")


# 主实验流程
def run_optimizer_comparison():
    """运行优化器比较实验"""
    print(f"\n{'=' * 80}")
    print(f"开始优化器比较实验")
    print(f"{'=' * 80}")

    results_list = []

    # 选择要比较的优化器（可以根据需要调整）
    selected_optimizers = ['Adam', 'SGD', 'RMSprop', 'AdaBelief', 'AccSGD', 'RAdam']

    for optimizer_name in selected_optimizers:
        try:
            print(f"\n正在训练: {optimizer_name}")
            result = train_model(optimizer_name, epochs=1000)
            if result is not None:
                results_list.append(result)
        except Exception as e:
            print(f"训练 {optimizer_name} 时出错: {e}")
            continue

    if not results_list:
        print("没有成功训练的模型！")
        return

    # 创建实验目录
    exp_dir = save_results(results_list, 'optimizer_comparison', save_dir, timestamp)

    # 生成对比图表
    print(f"\n生成对比图表...")
    plot_comparison_results(results_list, exp_dir)
    plot_membership_evolution_comparison(results_list, exp_dir)

    # 保存模糊规则
    print(f"\n保存模糊规则...")
    save_fuzzy_rules(results_list, exp_dir)

    # 生成总结报告
    generate_summary_report(results_list, exp_dir)

    print(f"\n实验完成！结果保存在: {exp_dir}")

    return results_list, exp_dir


def generate_summary_report(results_list, save_dir):
    """生成总结报告"""
    report_file = os.path.join(save_dir, f'summary_report_{timestamp}.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SANFIS优化器比较实验总结报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"实验时间: {timestamp}\n")
        f.write(f"数据集: 脱硫数据\n")
        f.write(f"输入特征: {', '.join(input_features)}\n")
        f.write(f"输出特征: {output_feature}\n\n")

        f.write("实验结果:\n")
        f.write("-" * 40 + "\n")

        # 按R²排序
        sorted_results = sorted(results_list, key=lambda x: x['r2'], reverse=True)

        for i, result in enumerate(sorted_results):
            f.write(f"{i + 1}. {result['optimizer_name']}\n")
            f.write(f"   MSE: {result['mse']:.6f}\n")
            f.write(f"   R²:  {result['r2']:.6f}\n\n")

        f.write(f"最佳优化器: {sorted_results[0]['optimizer_name']} ")
        f.write(f"(R² = {sorted_results[0]['r2']:.6f})\n")

        # 性能改进分析
        best_r2 = sorted_results[0]['r2']
        worst_r2 = sorted_results[-1]['r2']
        improvement = ((best_r2 - worst_r2) / worst_r2) * 100
        f.write(f"性能改进: {improvement:.2f}%\n")

    print(f"总结报告已保存: {report_file}")


# 运行实验
if __name__ == "__main__":
    results_list, exp_dir = run_optimizer_comparison()

    # 显示最佳结果
    if results_list:
        best_result = max(results_list, key=lambda x: x['r2'])
        print(f"\n{'=' * 60}")
        print(f"最佳结果:")
        print(f"优化器: {best_result['optimizer_name']}")
        print(f"MSE: {best_result['mse']:.6f}")
        print(f"R²: {best_result['r2']:.6f}")
        print(f"{'=' * 60}")
