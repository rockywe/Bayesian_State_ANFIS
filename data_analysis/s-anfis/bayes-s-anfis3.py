import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from itertools import product

from sanfis import SANFIS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
save_dir = 'bsanfis_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 加载数据
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

# 8个输入变量
feature_names = [
    'Gas_Inlet_Flow',
    'Gas_Inlet_Temperature',
    'Gas_Inlet_Pressure',
    'Desulfurization_Liquid_Flow',
    'Desulfurization_Liquid_Temperature',
    'Desulfurization_Liquid_Pressure',
    'Rotation_Speed',
    'H2S_Inlet_Concentration'
]
output_feature = 'H2S_Outlet_Concentration'

print(f"使用的8个输入变量: {feature_names}")
print(f"输出变量: {output_feature}")

# 数据清理和标准化
data_clean = data_df[feature_names + [output_feature]].dropna()
print(f"清理后的数据形状: {data_clean.shape}")

X_full = data_clean[feature_names].values
y_full = data_clean[output_feature].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_full)
y_scaled = scaler_y.fit_transform(y_full)

# 全局数据，供贝叶斯优化使用
X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_scaled_tensor = torch.tensor(y_scaled, dtype=torch.float32)

print(f"数据预处理完成: X shape = {X_scaled.shape}, y shape = {y_scaled.shape}")
# ================================
# 贝叶斯优化搜索空间定义
# ================================

# 为8个特征定义角色：每个特征可以是 'none', 'state', 'explanatory', 'both'
roles = ['state', 'explanatory', 'both']

# 构建搜索空间
space = []
for feature in feature_names:
    # 处理特征名，确保变量名合法
    safe_feature_name = feature.replace('_', '').replace('2', 'two')
    space.append(Categorical(roles, name=f'role_{safe_feature_name}'))

# 隶属函数数量 M (所有状态变量共享)
space.append(Integer(2, 4, name='M'))  # 控制在较小范围内避免规则爆炸

print(f"贝叶斯优化搜索空间维度: {len(space)}")
print(f"理论搜索空间大小: {4 ** 8 * 3} = {4 ** 8 * 3:,}")


# ================================
# SANFIS训练辅助函数
# ================================

def create_membfuncs_config(n_state_features, M):
    """为状态变量创建隶属函数配置"""
    membfuncs_config = []
    for i in range(n_state_features):
        mu_values = np.linspace(0.1, 0.9, M).tolist()
        sigma_values = [0.2] * M

        membfuncs_config.append({
            'function': 'gaussian',
            'n_memb': M,
            'params': {
                'mu': {'value': mu_values, 'trainable': True},
                'sigma': {'value': sigma_values, 'trainable': True}
            }
        })
    return membfuncs_config

# ================================
# 增强的SANFIS训练函数，返回更多指标
# ================================

def train_sanfis_model(X_s_train, X_x_train, y_train, X_s_val, X_x_val, y_val, M, epochs=200):
    """
    训练SANFIS模型，返回多个评价指标

    返回:
    - metrics: 包含RMSE, R², MAE等指标的字典
    """
    try:
        n_state_features = X_s_train.shape[1]
        n_expl_features = X_x_train.shape[1]

        # 创建隶属函数配置
        membfuncs_config = create_membfuncs_config(n_state_features, M)

        # 初始化SANFIS模型
        model = SANFIS(
            membfuncs=membfuncs_config,
            n_input=n_state_features,
            to_device='cpu',
            scale='Std'
        )

        # 设置优化器
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        loss_function = torch.nn.MSELoss(reduction='mean')

        # 训练模型
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            train_pred = model(X_s_train, X_x_train)
            train_loss = loss_function(train_pred, y_train)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 最终验证 - 计算多个指标
        model.eval()
        with torch.no_grad():
            val_pred = model(X_s_val, X_x_val)

            # 转换为numpy进行指标计算
            y_val_np = y_val.detach().numpy()
            val_pred_np = val_pred.detach().numpy()

            # 计算多个评价指标
            mse = mean_squared_error(y_val_np, val_pred_np)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val_np, val_pred_np)
            mae = mean_absolute_error(y_val_np, val_pred_np)

            # 计算平均绝对百分比误差 (MAPE)
            mape = np.mean(np.abs((y_val_np - val_pred_np) / (y_val_np + 1e-8))) * 100

            metrics = {
                'rmse': rmse,
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'mape': mape
            }

        return metrics

    except Exception as e:
        print(f"      SANFIS训练错误: {e}")
        return {
            'rmse': 1e10,
            'r2': -1e10,
            'mse': 1e10,
            'mae': 1e10,
            'mape': 1e10
        }


# ================================
# 增强的目标函数，记录更多信息
# ================================

# 全局变量用于记录优化过程
optimization_history = []


@use_named_args(space)
def objective(**params):
    """
    增强的BSANFIS贝叶斯优化目标函数
    """

    # 1. 解析参数
    state_features = []
    explanatory_features = []

    for feature in feature_names:
        safe_feature_name = feature.replace('_', '').replace('2', 'two')
        role = params[f'role_{safe_feature_name}']

        if role == 'state':
            state_features.append(feature)
        elif role == 'explanatory':
            explanatory_features.append(feature)
        elif role == 'both':
            state_features.append(feature)
            explanatory_features.append(feature)

    M = params['M']

    # 去重并排序
    state_features = sorted(list(set(state_features)))
    explanatory_features = sorted(list(set(explanatory_features)))

    print(f"\n{'=' * 60}")
    print(f"评估 #{len(optimization_history) + 1}")
    print(f"状态变量 (s): {state_features}")
    print(f"解释变量 (x): {explanatory_features}")
    print(f"隶属函数数量 (M): {M}")

    # 2. 边界条件检查
    if not state_features:
        print("跳过: 没有选择状态变量")
        return 1e10

    if not explanatory_features:
        print("跳过: 没有选择解释变量")
        return 1e10

    Ns = len(state_features)
    num_rules = M ** Ns
    MAX_RULES = 200

    if num_rules > MAX_RULES:
        print(f"跳过: 规则数量过多 ({num_rules} > {MAX_RULES})")
        return 1e10

    print(f"规则数量: {num_rules}")

    # 3. 准备数据
    try:
        state_indices = [feature_names.index(f) for f in state_features]
        expl_indices = [feature_names.index(f) for f in explanatory_features]

        X_s = X_scaled_tensor[:, state_indices]
        X_x = X_scaled_tensor[:, expl_indices]
        y = y_scaled_tensor

        print(f"数据形状: S={X_s.shape}, X={X_x.shape}, y={y.shape}")

    except Exception as e:
        print(f"数据准备错误: {e}")
        return 1e10

    # 4. K折交叉验证，记录所有指标
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_metrics = {
        'rmse': [], 'r2': [], 'mse': [], 'mae': [], 'mape': []
    }

    fold = 0
    for train_index, val_index in kf.split(X_s):
        fold += 1
        print(f"  训练第 {fold}/3 折...")

        try:
            # 划分数据
            X_s_train = X_s[train_index]
            X_x_train = X_x[train_index]
            y_train = y[train_index]

            X_s_val = X_s[val_index]
            X_x_val = X_x[val_index]
            y_val = y[val_index]

            # 训练SANFIS模型并获取所有指标
            metrics = train_sanfis_model(
                X_s_train, X_x_train, y_train,
                X_s_val, X_x_val, y_val,
                M, epochs=200
            )

            # 记录各项指标
            for metric_name, value in metrics.items():
                fold_metrics[metric_name].append(value)

            print(f"    第{fold}折 - RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")

        except Exception as e:
            print(f"    第{fold}折训练失败: {e}")
            for metric_name in fold_metrics:
                fold_metrics[metric_name].append(1e10 if metric_name != 'r2' else -1e10)

    # 计算平均指标
    avg_metrics = {}
    for metric_name, values in fold_metrics.items():
        avg_metrics[metric_name] = np.mean(values)
        avg_metrics[f'{metric_name}_std'] = np.std(values)

    print(f"平均指标:")
    print(f"  RMSE: {avg_metrics['rmse']:.6f} ± {avg_metrics['rmse_std']:.6f}")
    print(f"  R²: {avg_metrics['r2']:.6f} ± {avg_metrics['r2_std']:.6f}")
    print(f"  MAE: {avg_metrics['mae']:.6f} ± {avg_metrics['mae_std']:.6f}")

    # 记录这次评估的完整信息
    evaluation_record = {
        'evaluation_id': len(optimization_history) + 1,
        'state_features': state_features,
        'explanatory_features': explanatory_features,
        'M': M,
        'num_rules': num_rules,
        'num_state_vars': len(state_features),
        'num_expl_vars': len(explanatory_features),
        'metrics': avg_metrics,
        'fold_metrics': fold_metrics
    }
    optimization_history.append(evaluation_record)

    print(f"{'=' * 60}")

    # 返回RMSE作为优化目标（最小化）
    return avg_metrics['rmse']



# ================================
# 运行贝叶斯优化
# ================================

def run_bayesian_optimization():
    """运行BSANFIS的贝叶斯优化"""
    print(f"\n{'=' * 80}")
    print(f"开始BSANFIS贝叶斯优化")
    print(f"{'=' * 80}")
    print(f"搜索空间: 8个特征角色 + 1个隶属函数数量")
    print(f"优化目标: 最小化交叉验证RMSE")
    print(f"评估次数: 50次")
    print(f"交叉验证: 3折")
    print(f"每折训练轮数: 200")

    # 运行贝叶斯优化
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=50,  # 评估次数，可以根据时间预算调整
        n_initial_points=10,  # 初始随机点数量
        random_state=42,
        n_jobs=1,  # 串行执行，避免资源竞争
        acq_func='EI'  # 期望改进采集函数
    )

    return result

# ================================
# 可视化和分析函数
# ================================
def plot_comprehensive_analysis(optimization_result, optimization_history, save_dir):
    """绘制全面的分析图表"""

    # 创建一个大的图表
    fig = plt.figure(figsize=(20, 16))

    # 1. 优化收敛历史 (2x2的左上)
    ax1 = plt.subplot(3, 3, 1)
    rmse_values = [record['metrics']['rmse'] for record in optimization_history]
    r2_values = [record['metrics']['r2'] for record in optimization_history]

    ax1.plot(rmse_values, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax1.set_title('RMSE优化历史', fontsize=12, fontweight='bold')
    ax1.set_xlabel('评估次数')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)

    # 2. R²优化历史
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(r2_values, 'r-', linewidth=2, marker='s', markersize=4, alpha=0.7)
    ax2.set_title('R²优化历史', fontsize=12, fontweight='bold')
    ax2.set_xlabel('评估次数')
    ax2.set_ylabel('R²')
    ax2.grid(True, alpha=0.3)

    # 3. 最佳值变化
    ax3 = plt.subplot(3, 3, 3)
    best_rmse = np.minimum.accumulate(rmse_values)
    best_r2 = np.maximum.accumulate(r2_values)

    ax3_twin = ax3.twinx()
    line1 = ax3.plot(best_rmse, 'b-', linewidth=2, label='最佳RMSE')
    line2 = ax3_twin.plot(best_r2, 'r-', linewidth=2, label='最佳R²')

    ax3.set_xlabel('评估次数')
    ax3.set_ylabel('最佳RMSE', color='b')
    ax3_twin.set_ylabel('最佳R²', color='r')
    ax3.set_title('最佳指标变化', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. 特征选择频率统计
    ax4 = plt.subplot(3, 3, 4)
    feature_state_count = {f: 0 for f in feature_names}
    feature_expl_count = {f: 0 for f in feature_names}

    for record in optimization_history:
        for f in record['state_features']:
            feature_state_count[f] += 1
        for f in record['explanatory_features']:
            feature_expl_count[f] += 1

    features = list(feature_names)
    state_counts = [feature_state_count[f] for f in features]
    expl_counts = [feature_expl_count[f] for f in features]

    x = np.arange(len(features))
    width = 0.35

    ax4.bar(x - width / 2, state_counts, width, label='状态变量', alpha=0.8, color='skyblue')
    ax4.bar(x + width / 2, expl_counts, width, label='解释变量', alpha=0.8, color='lightcoral')

    ax4.set_xlabel('特征')
    ax4.set_ylabel('选择频次')
    ax4.set_title('特征选择频率统计', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f.replace('_', '\n') for f in features], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 隶属函数数量分布
    ax5 = plt.subplot(3, 3, 5)
    M_values = [record['M'] for record in optimization_history]
    M_counts = {m: M_values.count(m) for m in set(M_values)}

    ax5.bar(M_counts.keys(), M_counts.values(), alpha=0.8, color='lightgreen')
    ax5.set_xlabel('隶属函数数量 (M)')
    ax5.set_ylabel('选择频次')
    ax5.set_title('隶属函数数量分布', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. 规则数量 vs 性能
    ax6 = plt.subplot(3, 3, 6)
    rule_counts = [record['num_rules'] for record in optimization_history]

    scatter = ax6.scatter(rule_counts, rmse_values, c=r2_values, cmap='viridis', alpha=0.7, s=50)
    ax6.set_xlabel('规则数量')
    ax6.set_ylabel('RMSE')
    ax6.set_title('规则数量 vs 性能', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('R²')

    # 7. 状态变量数量 vs 性能
    ax7 = plt.subplot(3, 3, 7)
    state_var_counts = [record['num_state_vars'] for record in optimization_history]

    ax7.scatter(state_var_counts, rmse_values, alpha=0.7, color='orange', s=50)
    ax7.set_xlabel('状态变量数量')
    ax7.set_ylabel('RMSE')
    ax7.set_title('状态变量数量 vs RMSE', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # 8. 解释变量数量 vs 性能
    ax8 = plt.subplot(3, 3, 8)
    expl_var_counts = [record['num_expl_vars'] for record in optimization_history]

    ax8.scatter(expl_var_counts, r2_values, alpha=0.7, color='purple', s=50)
    ax8.set_xlabel('解释变量数量')
    ax8.set_ylabel('R²')
    ax8.set_title('解释变量数量 vs R²', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # 9. 性能指标相关性
    ax9 = plt.subplot(3, 3, 9)
    mae_values = [record['metrics']['mae'] for record in optimization_history]

    ax9.scatter(rmse_values, r2_values, alpha=0.7, color='red', s=50)
    ax9.set_xlabel('RMSE')
    ax9.set_ylabel('R²')
    ax9.set_title('RMSE vs R²', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)

    # 添加相关系数
    correlation = np.corrcoef(rmse_values, r2_values)[0, 1]
    ax9.text(0.05, 0.95, f'相关系数: {correlation:.3f}',
             transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comprehensive_analysis_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def train_final_bsanfis_model(optimization_result):
    """使用最优配置训练最终的BSANFIS模型"""
    print(f"\n{'=' * 80}")
    print(f"使用最优配置训练最终BSANFIS模型")
    print(f"{'=' * 80}")

    # 提取最优配置
    state_features = optimization_result['optimal_state_features']
    expl_features = optimization_result['optimal_expl_features']
    M = optimization_result['optimal_M']

    print(f"最优配置:")
    print(f"  状态变量: {state_features}")
    print(f"  解释变量: {expl_features}")
    print(f"  隶属函数数量: {M}")

    # 准备数据
    state_indices = [feature_names.index(f) for f in state_features]
    expl_indices = [feature_names.index(f) for f in expl_features]

    X_s = X_scaled_tensor[:, state_indices]
    X_x = X_scaled_tensor[:, expl_indices]
    y = y_scaled_tensor

    # 划分训练测试集
    train_indices, test_indices = train_test_split(
        range(len(X_s)), test_size=0.2, random_state=42
    )

    X_s_train = X_s[train_indices]
    X_x_train = X_x[train_indices]
    y_train = y[train_indices]

    X_s_test = X_s[test_indices]
    X_x_test = X_x[test_indices]
    y_test = y[test_indices]

    # 创建和训练最终模型
    n_state_features = len(state_features)
    membfuncs_config = create_membfuncs_config(n_state_features, M)

    final_model = SANFIS(
        membfuncs=membfuncs_config,
        n_input=n_state_features,
        to_device='cpu',
        scale='Std'
    )

    optimizer = torch.optim.RMSprop(final_model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss(reduction='mean')

    print(f"\n开始训练最终模型 (1000轮)...")

    # 训练历史记录
    train_losses = []
    val_losses = []

    final_model.train()
    for epoch in range(1000):
        optimizer.zero_grad()

        train_pred = final_model(X_s_train, X_x_train)
        train_loss = loss_function(train_pred, y_train)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录损失
        if epoch % 50 == 0:
            final_model.eval()
            with torch.no_grad():
                val_pred = final_model(X_s_test, X_x_test)
                val_loss = loss_function(val_pred, y_test)

                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())

                if epoch % 200 == 0:
                    print(f"  Epoch {epoch}: Train Loss = {train_loss.item():.6f}, "
                          f"Val Loss = {val_loss.item():.6f}")
            final_model.train()

    # 最终评估
    final_model.eval()
    with torch.no_grad():
        y_pred_scaled = final_model(X_s_test, X_x_test)

    # 反标准化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.detach().numpy())
    y_test_original = scaler_y.inverse_transform(y_test.detach().numpy())

    # 计算性能指标
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)
    mae = np.mean(np.abs(y_test_original - y_pred))

    print(f"\n最终模型性能:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")

    # 保存最终模型
    model_save_path = os.path.join(save_dir, f'final_bsanfis_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'state_features': state_features,
        'expl_features': expl_features,
        'membfuncs_config': membfuncs_config,
        'optimization_result': optimization_result,
        'performance': {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2
        }
    }, model_save_path)

    print(f"\n最终BSANFIS模型已保存: {model_save_path}")

    return {
        'model': final_model,
        'performance': {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2},
        'y_pred': y_pred,
        'y_test': y_test_original,
        'save_path': model_save_path
    }

def create_performance_summary_table(optimization_history, save_dir):
    """创建性能总结表"""

    # 找到最佳配置（前10个）
    sorted_history = sorted(optimization_history, key=lambda x: x['metrics']['rmse'])
    top_10 = sorted_history[:10]

    # 创建DataFrame
    summary_data = []
    for i, record in enumerate(top_10):
        summary_data.append({
            '排名': i + 1,
            '评估ID': record['evaluation_id'],
            'RMSE': f"{record['metrics']['rmse']:.6f}",
            'R²': f"{record['metrics']['r2']:.6f}",
            'MAE': f"{record['metrics']['mae']:.6f}",
            'MAPE(%)': f"{record['metrics']['mape']:.2f}",
            '状态变量数': record['num_state_vars'],
            '解释变量数': record['num_expl_vars'],
            '隶属函数数': record['M'],
            '规则总数': record['num_rules'],
            '状态变量': ', '.join(record['state_features'][:2]) + ('...' if len(record['state_features']) > 2 else ''),
            '解释变量': ', '.join(record['explanatory_features'][:2]) + (
                '...' if len(record['explanatory_features']) > 2 else '')
        })

    df = pd.DataFrame(summary_data)

    # 保存为CSV
    csv_path = os.path.join(save_dir, f'performance_summary_{timestamp}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 创建可视化表格
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # 只显示主要列
    display_columns = ['排名', 'RMSE', 'R²', 'MAE', '状态变量数', '解释变量数', '隶属函数数', '规则总数']
    display_df = df[display_columns]

    table = ax.table(cellText=display_df.values, colLabels=display_columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 设置表格样式
    for i in range(len(display_columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 高亮最佳结果
    for i in range(len(display_columns)):
        table[(1, i)].set_facecolor('#E8F5E8')

    plt.title('BSANFIS优化结果 - 前10名性能总结', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, f'performance_table_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"性能总结表已保存: {csv_path}")
    return df


def analyze_feature_importance(optimization_history, save_dir):
    """分析特征重要性"""

    # 计算每个特征作为状态变量和解释变量的平均性能
    feature_analysis = {}

    for feature in feature_names:
        # 作为状态变量时的性能
        state_performances = []
        expl_performances = []

        for record in optimization_history:
            if feature in record['state_features']:
                state_performances.append(record['metrics']['r2'])
            if feature in record['explanatory_features']:
                expl_performances.append(record['metrics']['r2'])

        feature_analysis[feature] = {
            'state_avg_r2': np.mean(state_performances) if state_performances else 0,
            'state_count': len(state_performances),
            'expl_avg_r2': np.mean(expl_performances) if expl_performances else 0,
            'expl_count': len(expl_performances),
            'total_appearances': len(state_performances) + len(expl_performances)
        }

    # 创建特征重要性图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    features = list(feature_names)

    # 1. 特征出现频次
    total_counts = [feature_analysis[f]['total_appearances'] for f in features]
    state_counts = [feature_analysis[f]['state_count'] for f in features]
    expl_counts = [feature_analysis[f]['expl_count'] for f in features]

    x = np.arange(len(features))
    width = 0.35

    ax1.bar(x - width / 2, state_counts, width, label='状态变量', alpha=0.8)
    ax1.bar(x + width / 2, expl_counts, width, label='解释变量', alpha=0.8)
    ax1.set_xlabel('特征')
    ax1.set_ylabel('出现频次')
    ax1.set_title('特征使用频次分析', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace('_', '\n') for f in features], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 作为状态变量时的平均R²
    state_r2s = [feature_analysis[f]['state_avg_r2'] for f in features]
    bars2 = ax2.bar(features, state_r2s, alpha=0.8, color='skyblue')
    ax2.set_xlabel('特征')
    ax2.set_ylabel('平均R²')
    ax2.set_title('作为状态变量时的平均性能', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, r2 in zip(bars2, state_r2s):
        if r2 > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{r2:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. 作为解释变量时的平均R²
    expl_r2s = [feature_analysis[f]['expl_avg_r2'] for f in features]
    bars3 = ax3.bar(features, expl_r2s, alpha=0.8, color='lightcoral')
    ax3.set_xlabel('特征')
    ax3.set_ylabel('平均R²')
    ax3.set_title('作为解释变量时的平均性能', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, r2 in zip(bars3, expl_r2s):
        if r2 > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{r2:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. 特征重要性综合评分
    # 综合评分 = 出现频次权重 * 0.3 + 平均性能权重 * 0.7
    importance_scores = []
    for f in features:
        freq_score = feature_analysis[f]['total_appearances'] / len(optimization_history)
        perf_score = max(feature_analysis[f]['state_avg_r2'], feature_analysis[f]['expl_avg_r2'])
        combined_score = freq_score * 0.3 + perf_score * 0.7
        importance_scores.append(combined_score)

    # 排序
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_scores = [importance_scores[i] for i in sorted_indices]

    bars4 = ax4.bar(range(len(sorted_features)), sorted_scores, alpha=0.8, color='lightgreen')
    ax4.set_xlabel('特征（按重要性排序）')
    ax4.set_ylabel('综合重要性评分')
    ax4.set_title('特征重要性排名', fontweight='bold')
    ax4.set_xticks(range(len(sorted_features)))
    ax4.set_xticklabels([f.replace('_', '\n') for f in sorted_features], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, score in zip(bars4, sorted_scores):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'feature_importance_analysis_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 保存特征分析结果
    feature_analysis_df = pd.DataFrame.from_dict(feature_analysis, orient='index')
    feature_analysis_df['importance_score'] = importance_scores
    feature_analysis_df = feature_analysis_df.sort_values('importance_score', ascending=False)

    analysis_path = os.path.join(save_dir, f'feature_analysis_{timestamp}.csv')
    feature_analysis_df.to_csv(analysis_path, encoding='utf-8-sig')

    print(f"特征重要性分析已保存: {analysis_path}")
    return feature_analysis_df


# ================================
# 修改主函数，添加全面分析
# ================================

def analyze_optimization_result(result):
    """分析优化结果 - 增强版"""
    print(f"\n{'=' * 80}")
    print(f"贝叶斯优化结果分析")
    print(f"{'=' * 80}")

    print(f"最佳RMSE: {result.fun:.6f}")
    print(f"总评估次数: {len(result.func_vals)}")

    # 解析最佳参数
    best_params = dict(zip([s.name for s in space], result.x))

    optimal_state_features = []
    optimal_expl_features = []

    for feature in feature_names:
        safe_feature_name = feature.replace('_', '').replace('2', 'two')
        role = best_params[f'role_{safe_feature_name}']

        if role == 'state':
            optimal_state_features.append(feature)
        elif role == 'explanatory':
            optimal_expl_features.append(feature)
        elif role == 'both':
            optimal_state_features.append(feature)
            optimal_expl_features.append(feature)

    optimal_M = best_params['M']

    # 找到最佳配置的详细指标
    best_record = min(optimization_history, key=lambda x: x['metrics']['rmse'])

    print(f"\n最优配置:")
    print(f"状态变量 (s): {optimal_state_features}")
    print(f"解释变量 (x): {optimal_expl_features}")
    print(f"隶属函数数量 (M): {optimal_M}")
    print(f"规则总数: {optimal_M ** len(optimal_state_features)}")

    print(f"\n最优配置的详细性能指标:")
    print(f"RMSE: {best_record['metrics']['rmse']:.6f} ± {best_record['metrics']['rmse_std']:.6f}")
    print(f"R²: {best_record['metrics']['r2']:.6f} ± {best_record['metrics']['r2_std']:.6f}")
    print(f"MAE: {best_record['metrics']['mae']:.6f} ± {best_record['metrics']['mae_std']:.6f}")
    print(f"MAPE: {best_record['metrics']['mape']:.2f}% ± {best_record['metrics']['mape_std']:.2f}%")

    # 保存结果
    optimization_result = {
        'best_rmse': result.fun,
        'best_params': best_params,
        'optimal_state_features': optimal_state_features,
        'optimal_expl_features': optimal_expl_features,
        'optimal_M': optimal_M,
        'total_rules': optimal_M ** len(optimal_state_features),
        'best_detailed_metrics': best_record['metrics'],
        'optimization_history': optimization_history,
        'timestamp': timestamp,
        'feature_names': feature_names
    }

    # 保存到文件
    result_file = os.path.join(save_dir, f'bsanfis_optimization_result_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)

        json.dump(deep_convert(optimization_result), f, ensure_ascii=False, indent=2)

    print(f"\n优化结果已保存: {result_file}")

    return optimization_result


# ================================
# 修改主实验流程
# ================================

if __name__ == "__main__":
    print(f"开始BSANFIS实验 - 时间戳: {timestamp}")

    # 第一阶段：贝叶斯优化寻找最优结构
    print(f"\n第一阶段：贝叶斯优化寻找最优结构")
    optimization_result = run_bayesian_optimization()

    # 分析优化结果
    result_analysis = analyze_optimization_result(optimization_result)

    # 全面的可视化分析
    print(f"\n生成全面分析图表...")
    plot_comprehensive_analysis(optimization_result, optimization_history, save_dir)

    # 创建性能总结表
    print(f"\n创建性能总结表...")
    performance_df = create_performance_summary_table(optimization_history, save_dir)

    # 特征重要性分析
    print(f"\n进行特征重要性分析...")
    feature_importance_df = analyze_feature_importance(optimization_history, save_dir)

    # 第二阶段：使用最优配置训练最终模型
    print(f"\n第二阶段：使用最优配置训练最终模型")
    final_model_result = train_final_bsanfis_model(result_analysis)

    print(f"\n{'=' * 80}")
    print(f"BSANFIS实验完成！")
    print(f"{'=' * 80}")
    print(f"贝叶斯优化找到的最优配置:")
    print(f"  状态变量: {result_analysis['optimal_state_features']}")
    print(f"  解释变量: {result_analysis['optimal_expl_features']}")
    print(f"  隶属函数数量: {result_analysis['optimal_M']}")
    print(f"  模糊规则总数: {result_analysis['total_rules']}")
    print(f"\n优化过程中的最佳指标:")
    print(f"  RMSE: {result_analysis['best_detailed_metrics']['rmse']:.6f}")
    print(f"  R²: {result_analysis['best_detailed_metrics']['r2']:.6f}")
    print(f"  MAE: {result_analysis['best_detailed_metrics']['mae']:.6f}")
    print(f"\n最终模型性能:")
    print(f"  MSE: {final_model_result['performance']['mse']:.6f}")
    print(f"  RMSE: {final_model_result['performance']['rmse']:.6f}")
    print(f"  R²: {final_model_result['performance']['r2']:.6f}")
    print(f"  MAE: {final_model_result['performance']['mae']:.6f}")
    print(f"\n所有结果已保存到: {save_dir}")
