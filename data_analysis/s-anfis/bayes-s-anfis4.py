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
# ================================
# 全局设备配置
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
save_dir = 'sanfis_results/bsanfis_results'
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

def train_sanfis_model(X_s_train, X_x_train, y_train, X_s_val, X_x_val, y_val, M, epochs=1000):
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
        ).to(device)

        # 设置优化器
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        loss_function = torch.nn.MSELoss(reduction='mean')
        X_s_train, X_x_train, y_train = X_s_train.to(device), X_x_train.to(device), y_train.to(device)
        X_s_val, X_x_val, y_val = X_s_val.to(device), X_x_val.to(device), y_val.to(device)
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
            # y_val_np = y_val.detach().numpy()
            # val_pred_np = val_pred.detach().numpy()
            y_val_np = y_val.cpu().detach().numpy()
            val_pred_np = val_pred.cpu().detach().numpy()

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
# 增强的目标函数，带有对“转速”角色的引导
# ================================

# ================================
# 增强的目标函数，带有对“转速”角色的引导和更稳健的错误处理
# ================================

# 全局变量用于记录优化过程
optimization_history = []


@use_named_args(space)
def objective(**params):
    """
    增强的BSANFIS贝叶斯优化目标函数。
    此版本包含一个“软约束”，并增加了更稳健的错误处理以避免NaN。
    """
    try:
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
        state_features = sorted(list(set(state_features)))
        explanatory_features = sorted(list(set(explanatory_features)))

        print(f"\n{'=' * 60}")
        print(f"评估 #{len(optimization_history) + 1}")
        print(f"状态变量 (s): {state_features}")
        print(f"解释变量 (x): {explanatory_features}")
        print(f"隶属函数数量 (M): {M}")

        # 2. 边界条件检查
        if not state_features or not explanatory_features:
            print("跳过: 缺少状态或解释变量")
            return 1e10
        Ns = len(state_features)
        num_rules = M ** Ns
        MAX_RULES = 2000
        if num_rules > MAX_RULES:
            print(f"跳过: 规则数量过多 ({num_rules} > {MAX_RULES})")
            return 1e10
        print(f"规则数量: {num_rules}")

        # 3. 准备数据
        state_indices = [feature_names.index(f) for f in state_features]
        expl_indices = [feature_names.index(f) for f in explanatory_features]
        X_s = X_scaled_tensor[:, state_indices]
        X_x = X_scaled_tensor[:, expl_indices]
        y = y_scaled_tensor

        # 4. K折交叉验证
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_metrics = {'rmse': [], 'r2': [], 'mse': [], 'mae': [], 'mape': []}

        for fold, (train_index, val_index) in enumerate(kf.split(X_s)):
            print(f"  训练第 {fold + 1}/3 折...")
            try:
                X_s_train, X_x_train, y_train = X_s[train_index], X_x[train_index], y[train_index]
                X_s_val, X_x_val, y_val = X_s[val_index], X_x[val_index], y[val_index]

                # 增加一个检查，防止y_val中所有值都相同
                if torch.unique(y_val).numel() < 2:
                    print("    警告: 验证集目标值全部相同，跳过此折评估。")
                    # 给予一个中等程度的惩罚，因为它不是一个好的划分
                    metrics = {'rmse': 1e5, 'r2': 0.0, 'mse': 1e10, 'mae': 1e5, 'mape': 1e5}
                else:
                    metrics = train_sanfis_model(
                        X_s_train, X_x_train, y_train,
                        X_s_val, X_x_val, y_val,
                        M, epochs=1000
                    )

                # 再次检查返回的metrics是否包含NaN
                if np.isnan(metrics['rmse']):
                    print("    错误: train_sanfis_model返回了NaN的RMSE。使用惩罚值。")
                    metrics['rmse'] = 1e10

                for metric_name, value in metrics.items():
                    fold_metrics[metric_name].append(value)

                print(f"    第{fold + 1}折 - RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")

            except Exception as e:
                print(f"    第{fold + 1}折训练时发生严重错误: {e}")
                # 如果发生任何未预料的错误，都添加一个大的惩罚值
                for metric_name in fold_metrics:
                    fold_metrics[metric_name].append(1e10 if metric_name != 'r2' else -1e10)

        # 5. 计算平均指标，并处理可能存在的NaN（作为最后一道防线）
        avg_metrics = {}
        for name, values in fold_metrics.items():
            # 使用np.nanmean可以安全地忽略NaN值进行计算
            avg_metrics[name] = np.nanmean(values)
            avg_metrics[f'{name}_std'] = np.nanstd(values)

        # 如果计算后仍然是NaN（例如所有折都失败了），则给予最终惩罚
        if np.isnan(avg_metrics['rmse']):
            print("错误: 所有折都失败，无法计算平均RMSE。返回最终惩罚值。")
            return 1e12

        print(f"平均指标 (无惩罚):")
        print(f"  RMSE: {avg_metrics['rmse']:.6f}, R²: {avg_metrics['r2']:.6f}")

        # 6. 软约束引导
        penalty_factor = 1.0
        rotation_speed_role = params['role_RotationSpeed']
        if rotation_speed_role != 'both':
            penalty_factor = 1.02
            print(f"  应用惩罚: 'Rotation_Speed'角色为'{rotation_speed_role}'而非'both'。惩罚因子: {penalty_factor}")
        else:
            print(f"  满足先验: 'Rotation_Speed'角色为'both'。无惩罚。")

        final_score_for_optimizer = avg_metrics['rmse'] * penalty_factor
        print(f"  最终优化分数 (RMSE * 惩罚因子): {final_score_for_optimizer:.6f}")

        # 7. 记录历史
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
        return final_score_for_optimizer

    except Exception as e:
        # 这是一个全局的try-except，捕获任何在函数顶层发生的意外错误
        print(f"在objective函数顶层捕获到严重错误: {e}")
        # 返回一个巨大的惩罚值，确保不会是NaN
        return 1e12


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
        n_initial_points=20,  # 初始随机点数量
        random_state=42,
        n_jobs=2,  # 串行执行，避免资源竞争
        acq_func='EI'  # 期望改进采集函数
    )

    return result

# ================================
# 可视化和分析函数
# ================================


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

    print(f"\n开始训练最终模型 (2000轮)...")

    # 训练历史记录
    train_losses = []
    val_losses = []

    final_model.train()
    for epoch in range(2000):
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


def analyze_feature_importance(optimization_history, save_dir):
    """分析特征重要性，并确保所有评分为正数"""

    feature_analysis = {}

    for feature in feature_names:
        state_performances = [record['metrics']['r2'] for record in optimization_history if
                              feature in record['state_features']]
        expl_performances = [record['metrics']['r2'] for record in optimization_history if
                             feature in record['explanatory_features']]

        # 使用max(0, r2)来计算平均性能，确保结果非负
        state_avg_r2 = np.mean([max(0, r2) for r2 in state_performances]) if state_performances else 0
        expl_avg_r2 = np.mean([max(0, r2) for r2 in expl_performances]) if expl_performances else 0

        feature_analysis[feature] = {
            'state_avg_r2': state_avg_r2,
            'state_count': len(state_performances),
            'expl_avg_r2': expl_avg_r2,
            'expl_count': len(expl_performances),
            'total_appearances': len(state_performances) + len(expl_performances)
        }

    # 创建特征重要性图 (这部分可以保留，因为它提供了详细信息)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    features = list(feature_names)
    x = np.arange(len(features))
    width = 0.35

    # 1. 特征出现频次
    state_counts = [feature_analysis[f]['state_count'] for f in features]
    expl_counts = [feature_analysis[f]['expl_count'] for f in features]
    ax1.bar(x - width / 2, state_counts, width, label='s', alpha=0.8)
    ax1.bar(x + width / 2, expl_counts, width, label='x', alpha=0.8)
    # ax1.set_title('特征使用频次分析', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace('_', '\n') for f in features], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. 作为状态变量时的平均R² (非负)
    state_r2s = [feature_analysis[f]['state_avg_r2'] for f in features]
    ax2.bar(features, state_r2s, alpha=0.8, color='skyblue')
    # ax2.set_title('作为状态变量时的平均性能 (非负R²)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 作为解释变量时的平均R² (非负)
    expl_r2s = [feature_analysis[f]['expl_avg_r2'] for f in features]
    ax3.bar(features, expl_r2s, alpha=0.8, color='lightcoral')
    # ax3.set_title('作为解释变量时的平均性能 (非负R²)', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 特征重要性综合评分 (确保为正)
    importance_scores = []
    for f in features:
        freq_score = feature_analysis[f]['total_appearances'] / len(optimization_history)
        # perf_score现在总是非负的
        perf_score = max(feature_analysis[f]['state_avg_r2'], feature_analysis[f]['expl_avg_r2'])
        combined_score = freq_score * 0.3 + perf_score * 0.7
        importance_scores.append(combined_score)

    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_scores = [importance_scores[i] for i in sorted_indices]
    ax4.bar(range(len(sorted_features)), sorted_scores, alpha=0.8, color='lightgreen')
    # ax4.set_title('特征重要性排名 (综合评分)', fontweight='bold')
    ax4.set_xticks(range(len(sorted_features)))
    ax4.set_xticklabels([f.replace('_', '\n') for f in sorted_features], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'feature_importance_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 保存特征分析结果
    feature_analysis_df = pd.DataFrame.from_dict(feature_analysis, orient='index')
    feature_analysis_df['importance_score'] = importance_scores
    feature_analysis_df = feature_analysis_df.sort_values('importance_score', ascending=False)
    analysis_path = os.path.join(save_dir, f'feature_analysis_{timestamp}.csv')
    feature_analysis_df.to_csv(analysis_path, encoding='utf-8-sig')
    print(f"特征重要性分析已保存: {analysis_path}")
    return feature_analysis_df


def plot_key_analysis(optimization_history, save_dir):
    """
    绘制简化的关键分析图表，包含三个核心子图。
    R²值在绘图时被处理为非负。
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    sns.set_style("whitegrid")

    # --- 图1: 特征选择频率统计 ---
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
    width = 0.4

    ax1.bar(x - width / 2, state_counts, width, label='(s)', alpha=0.8, color='skyblue')
    ax1.bar(x + width / 2, expl_counts, width, label='(x)', alpha=0.8, color='lightcoral')

    ax1.set_ylabel('Select frequency', fontsize=12)
    # ax1.set_title('特征选择频率统计', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace('_', '\n') for f in features], rotation=60, ha='right', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.4, axis='y')

    # --- 图2: 隶属函数数量分布 ---
    M_values = [record['M'] for record in optimization_history]
    M_counts = {m: M_values.count(m) for m in sorted(list(set(M_values)))}

    ax2.bar(M_counts.keys(), M_counts.values(), alpha=0.8, color='lightgreen', width=0.6)
    ax2.set_xlabel('Number of membership functions (M)', fontsize=12)
    ax2.set_ylabel('Select frequency', fontsize=12)
    # ax2.set_title('隶属函数数量分布', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.4, axis='y')
    ax2.set_xticks(list(M_counts.keys()))  # 确保x轴刻度是整数

    # --- 图3: 规则数量 vs 性能 ---
    rule_counts = [record['num_rules'] for record in optimization_history]
    rmse_values = [record['metrics']['rmse'] for record in optimization_history]

    # 将R²处理为非负值用于着色
    r2_values_for_plot = [max(0, record['metrics']['r2']) for record in optimization_history]

    scatter = ax3.scatter(rule_counts, rmse_values, c=r2_values_for_plot, cmap='viridis', alpha=0.7, s=60,
                          edgecolors='w')
    ax3.set_xlabel('Number of rules', fontsize=12)
    ax3.set_ylabel('RMSE', fontsize=12)
    # ax3.set_title('规则数量 vs 性能', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.4)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('R²', fontsize=12)

    # 调整整体布局
    plt.suptitle('BSANFIS 优化过程关键分析', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为总标题留出空间
    plt.savefig(os.path.join(save_dir, f'key_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.show()


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

    # --- 修改部分开始 ---
    # 调用新的、简化的可视化函数
    print(f"\n生成关键分析图表...")
    plot_key_analysis(optimization_history, save_dir)

    # （可选）可以保留或删除详细的特征重要性分析图
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
