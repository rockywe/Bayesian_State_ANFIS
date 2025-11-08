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

# <--- 修改: 1. 增加SHAP库导入
# 请确保已安装SHAP库: pip install shap
try:
    import shap
except ImportError:
    print("错误: SHAP库未安装。请运行 'pip install shap' 进行安装。")
    exit()
# ---> 修改结束

from sanfis import SANFIS

# ================================
# 全局设备配置
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
# 设置中文字体
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial','Helvetica', 'Helvetica World', 'Arial Unicode MS']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['path.simplify'] = True
plt.rcParams['path.snap'] = True

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

# 为8个特征定义角色：每个特征可以是 'state', 'explanatory', 'both'
roles = ['state', 'explanatory', 'both']

# 构建搜索空间
space = []
for feature in feature_names:
    safe_feature_name = feature.replace('_', '').replace('2', 'two')
    # <--- 修改: 2. 将转速的角色固定为 'both'，实现硬约束
    if feature == 'Rotation_Speed':
        space.append(Categorical(['both'], name=f'role_{safe_feature_name}'))
    else:
        # 其他变量保持可选
        space.append(Categorical(roles, name=f'role_{safe_feature_name}'))
    # ---> 修改结束

# 隶属函数数量 M (所有状态变量共享)
space.append(Integer(2, 4, name='M'))  # 控制在较小范围内避免规则爆炸

print(f"贝叶斯优化搜索空间维度: {len(space)}")


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

def train_sanfis_model(X_s_train, X_x_train, y_train, X_s_val, X_x_val, y_val, M, epochs=1500):
    """
    训练SANFIS模型，返回多个评价指标
    """
    try:
        n_state_features = X_s_train.shape[1]
        n_expl_features = X_x_train.shape[1]

        membfuncs_config = create_membfuncs_config(n_state_features, M)

        model = SANFIS(
            membfuncs=membfuncs_config,
            n_input=n_state_features,
            to_device='cpu',
            scale='Std'
        ).to(device)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        loss_function = torch.nn.MSELoss(reduction='mean')
        X_s_train, X_x_train, y_train = X_s_train.to(device), X_x_train.to(device), y_train.to(device)
        X_s_val, X_x_val, y_val = X_s_val.to(device), X_x_val.to(device), y_val.to(device)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            train_pred = model(X_s_train, X_x_train)
            train_loss = loss_function(train_pred, y_train)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_s_val, X_x_val)
            y_val_np = y_val.cpu().detach().numpy()
            val_pred_np = val_pred.cpu().detach().numpy()

            mse = mean_squared_error(y_val_np, val_pred_np)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val_np, val_pred_np)
            mae = mean_absolute_error(y_val_np, val_pred_np)
            mape = np.mean(np.abs((y_val_np - val_pred_np) / (y_val_np + 1e-8))) * 100

            metrics = {'rmse': rmse, 'r2': r2, 'mse': mse, 'mae': mae, 'mape': mape}

        return metrics

    except Exception as e:
        print(f"      SANFIS训练错误: {e}")
        return {'rmse': 1e10, 'r2': -1e10, 'mse': 1e10, 'mae': 1e10, 'mape': 1e10}


# ================================
# 目标函数，已移除软约束
# ================================
optimization_history = []


@use_named_args(space)
def objective(**params):
    """
    BSANFIS贝叶斯优化目标函数。
    引入R²作为引导，鼓励优化器寻找RMSE和R²双优的解。
    """
    try:
        # ... (函数前半部分的参数解析和边界检查保持不变) ...
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

        if not state_features or not explanatory_features:
            print("跳过: 缺少状态或解释变量")
            return 1e10
        if 'Rotation_Speed' not in state_features or 'Rotation_Speed' not in explanatory_features:
            print("严重错误: 'Rotation_Speed' 未被分配为 'both' 角色。返回高惩罚值。")
            return 1e12
        Ns = len(state_features)
        num_rules = M ** Ns
        MAX_RULES = 3000  # 配合M的增加，适当放宽规则数
        if num_rules > MAX_RULES:
            print(f"跳过: 规则数量过多 ({num_rules} > {MAX_RULES})")
            return 1e10
        print(f"规则数量: {num_rules}")

        state_indices = [feature_names.index(f) for f in state_features]
        expl_indices = [feature_names.index(f) for f in explanatory_features]
        X_s = X_scaled_tensor[:, state_indices]
        X_x = X_scaled_tensor[:, expl_indices]
        y = y_scaled_tensor

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_metrics = {'rmse': [], 'r2': [], 'mse': [], 'mae': [], 'mape': []}

        for fold, (train_index, val_index) in enumerate(kf.split(X_s)):
            print(f"  训练第 {fold + 1}/3 折...")
            try:
                X_s_train, X_x_train, y_train = X_s[train_index], X_x[train_index], y[train_index]
                X_s_val, X_x_val, y_val = X_s[val_index], X_x[val_index], y[val_index]

                if torch.unique(y_val).numel() < 2:
                    metrics = {'rmse': 1e5, 'r2': 0.0, 'mse': 1e10, 'mae': 1e5, 'mape': 1e5}
                else:
                    # <--- 修改: 2. 增加训练轮数
                    metrics = train_sanfis_model(
                        X_s_train, X_x_train, y_train,
                        X_s_val, X_x_val, y_val,
                        M, epochs=1500  # 从 1000 增加到 1500
                    )
                    # ---> 修改结束

                if np.isnan(metrics['rmse']): metrics['rmse'] = 1e10
                for metric_name, value in metrics.items(): fold_metrics[metric_name].append(value)
                print(f"    第{fold + 1}折 - RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")

            except Exception as e:
                print(f"    第{fold + 1}折训练时发生严重错误: {e}")
                for metric_name in fold_metrics: fold_metrics[metric_name].append(
                    1e10 if metric_name != 'r2' else -1e10)

        avg_metrics = {}
        for name, values in fold_metrics.items():
            avg_metrics[name] = np.nanmean(values)
            avg_metrics[f'{name}_std'] = np.nanstd(values)

        if np.isnan(avg_metrics['rmse']): return 1e12

        print(f"平均指标:")
        print(f"  RMSE: {avg_metrics['rmse']:.6f}, R²: {avg_metrics['r2']:.6f}")

        # <--- 修改: 3. 核心修改：引入R²引导机制
        # 使用R²对RMSE进行加权，鼓励高R²的解
        # (1.1 - R²) 作为惩罚/奖励因子。R²越高，因子越小，最终分数越低（越好）
        guidance_factor = 1.1 - avg_metrics['r2']
        # 确保因子不会因极端R²值（如<-0.1）变为负数，虽然不太可能
        guidance_factor = max(guidance_factor, 0.01)

        final_score_for_optimizer = avg_metrics['rmse'] * guidance_factor

        print(f"  引导因子 (1.1 - R²): {guidance_factor:.4f}")
        print(f"  最终优化分数 (RMSE * 引导因子): {final_score_for_optimizer:.6f}")
        # ---> 修改结束

        evaluation_record = {
            'evaluation_id': len(optimization_history) + 1,
            'state_features': state_features,
            'explanatory_features': explanatory_features,
            'M': M,
            'num_rules': num_rules,
            'num_state_vars': len(state_features),
            'num_expl_vars': len(explanatory_features),
            'metrics': avg_metrics,
            'fold_metrics': fold_metrics,
            'final_score': final_score_for_optimizer  # 记录最终分数
        }
        optimization_history.append(evaluation_record)

        print(f"{'=' * 60}")
        return final_score_for_optimizer

    except Exception as e:
        print(f"在objective函数顶层捕获到严重错误: {e}")
        return 1e12


# ================================
# 运行贝叶斯优化
# ================================

def run_bayesian_optimization():
    """运行BSANFIS的贝叶斯优化"""
    print(f"\n{'=' * 80}")
    print(f"开始BSANFIS贝叶斯优化")
    print(f"{'=' * 80}")
    print(f"搜索空间: {len(space)}个维度")
    print(f"约束: 'Rotation_Speed' 角色固定为 'both'")
    print(f"优化目标: 最小化交叉验证RMSE")
    print(f"评估次数: 50次")
    print(f"交叉验证: 3折")
    print(f"每折训练轮数: 1000")

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=50,
        n_initial_points=20,
        random_state=42,
        n_jobs=-1,  # 使用所有可用CPU核心
        acq_func='EI'
    )
    return result


# ================================
# 可视化和分析函数
# (这部分函数保持不变)
# ================================
def plot_key_analysis(optimization_history, save_dir):
    """
    绘制修改后的关键分析图表，包含两个核心子图。
    1. 隶属函数数量分布 (M)。
    2. 按规则数量分组的性能箱线图 (RMSE & R²)。
    此版本修复了从skopt.space获取M值范围的错误。
    """
    # 创建一个包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_style("whitegrid")

    # --- 图1: 隶属函数数量分布 (原中间的图) ---
    M_values = [record['M'] for record in optimization_history]

    # <--- 修改: 核心修复逻辑
    # 从 'space' 变量中正确地提取 M 的所有可能值
    all_possible_M = []
    for r in space:
        if hasattr(r, 'name') and r.name == 'M' and isinstance(r, Integer):
            all_possible_M = list(range(r.low, r.high + 1))
            break  # 找到M后就退出循环
    # ---> 修改结束

    # 确保即使某些M值未被选中，也能在图中显示
    M_counts = {m: M_values.count(m) for m in all_possible_M}

    ax1.bar(M_counts.keys(), M_counts.values(), alpha=0.8, color='lightgreen', width=0.6)
    ax1.set_xlabel('Number of membership functions (M)', fontsize=14)
    ax1.set_ylabel('Select Frequency', fontsize=14)
    ax1.set_title('Distribution of Membership Functions (M)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.5, axis='y')
    ax1.set_xticks(all_possible_M)  # 确保x轴刻度是整数
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # --- 图2: 按规则数量分组的性能箱线图 (新的、更直观的图) ---
    # (这部分代码保持不变)
    df_data = {
        'num_rules': [record['num_rules'] for record in optimization_history],
        'rmse': [record['metrics']['rmse'] for record in optimization_history],
        'r2': [record['metrics']['r2'] for record in optimization_history]
    }
    perf_df = pd.DataFrame(df_data)

    perf_df_filtered = perf_df[perf_df['rmse'] < 1.0].copy()

    # 检查过滤后的数据是否为空
    if perf_df_filtered.empty:
        print("警告: 在plot_key_analysis中，没有RMSE < 1.0的数据点，无法生成性能图。")
        # 可以选择画一个空图或者直接返回
        ax2.text(0.5, 0.5, 'No data to display\n(All RMSE >= 1.0)',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=12, color='red')
    else:
        bins = pd.cut(perf_df_filtered['num_rules'], bins=5, labels=False, duplicates='drop')
        perf_df_filtered['rule_bin'] = bins

        bin_labels = {}
        for i in sorted(perf_df_filtered['rule_bin'].unique()):
            min_val = perf_df_filtered[perf_df_filtered['rule_bin'] == i]['num_rules'].min()
            max_val = perf_df_filtered[perf_df_filtered['rule_bin'] == i]['num_rules'].max()
            bin_labels[i] = f'{min_val}-{max_val}'

        perf_df_filtered['rule_bin_label'] = perf_df_filtered['rule_bin'].map(bin_labels)

        sns.boxplot(x='rule_bin_label', y='rmse', data=perf_df_filtered, ax=ax2, palette="coolwarm")
        ax2.set_xlabel('Number of Rules (Grouped)', fontsize=14)
        ax2.set_ylabel('RMSE', fontsize=14, color='b')
        ax2.set_title('Performance by Number of Rules', fontsize=16, fontweight='bold')
        ax2.tick_params(axis='x', rotation=30, labelsize=12)
        ax2.tick_params(axis='y', labelsize=12, labelcolor='b')
        ax2.grid(True, alpha=0.5)

        ax2_twin = ax2.twinx()
        sns.stripplot(x='rule_bin_label', y='r2', data=perf_df_filtered, ax=ax2_twin,
                      color='purple', alpha=0.6, jitter=0.2, size=6)
        ax2_twin.set_ylabel('R²', fontsize=14, color='purple')
        ax2_twin.tick_params(axis='y', labelsize=12, labelcolor='purple')
        ax2_twin.set_ylim(0, 1.05)

    # 调整整体布局
    plt.suptitle('BSANFIS Optimization Process Analysis', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    # 保存图片
    save_path = os.path.join(save_dir, f'key_analysis_revised_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"新的关键分析图已保存到: {save_path}")


def analyze_feature_importance(optimization_history, save_dir):
    feature_analysis = {}
    for feature in feature_names:
        state_performances = [record['metrics']['r2'] for record in optimization_history if
                              feature in record['state_features']]
        expl_performances = [record['metrics']['r2'] for record in optimization_history if
                             feature in record['explanatory_features']]
        state_avg_r2 = np.mean([max(0, r2) for r2 in state_performances]) if state_performances else 0
        expl_avg_r2 = np.mean([max(0, r2) for r2 in expl_performances]) if expl_performances else 0
        feature_analysis[feature] = {
            'state_avg_r2': state_avg_r2,
            'state_count': len(state_performances),
            'expl_avg_r2': expl_avg_r2,
            'expl_count': len(expl_performances),
            'total_appearances': len(state_performances) + len(expl_performances)
        }
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    features = list(feature_names)
    x = np.arange(len(features))
    width = 0.35
    state_counts = [feature_analysis[f]['state_count'] for f in features]
    expl_counts = [feature_analysis[f]['expl_count'] for f in features]
    ax1.bar(x - width / 2, state_counts, width, label='s', alpha=0.8)
    ax1.bar(x + width / 2, expl_counts, width, label='x', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace('_', '\n') for f in features], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    state_r2s = [feature_analysis[f]['state_avg_r2'] for f in features]
    ax2.bar(features, state_r2s, alpha=0.8, color='skyblue')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    expl_r2s = [feature_analysis[f]['expl_avg_r2'] for f in features]
    ax3.bar(features, expl_r2s, alpha=0.8, color='lightcoral')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    importance_scores = []
    for f in features:
        freq_score = feature_analysis[f]['total_appearances'] / len(optimization_history)
        perf_score = max(feature_analysis[f]['state_avg_r2'], feature_analysis[f]['expl_avg_r2'])
        combined_score = freq_score * 0.3 + perf_score * 0.7
        importance_scores.append(combined_score)
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_scores = [importance_scores[i] for i in sorted_indices]
    ax4.bar(range(len(sorted_features)), sorted_scores, alpha=0.8, color='lightgreen')
    ax4.set_xticks(range(len(sorted_features)))
    ax4.set_xticklabels([f.replace('_', '\n') for f in sorted_features], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'feature_importance_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    feature_analysis_df = pd.DataFrame.from_dict(feature_analysis, orient='index')
    feature_analysis_df['importance_score'] = importance_scores
    feature_analysis_df = feature_analysis_df.sort_values('importance_score', ascending=False)
    analysis_path = os.path.join(save_dir, f'feature_analysis_{timestamp}.csv')
    feature_analysis_df.to_csv(analysis_path, encoding='utf-8-sig')
    print(f"特征重要性分析已保存: {analysis_path}")
    return feature_analysis_df


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
    # 使用一个安全的默认值，以防 optimization_history 为空
    if not optimization_history:
        print("警告: 优化历史为空，无法找到最佳记录。")
        best_record = {
            'metrics': {'rmse': float('inf'), 'r2': float('-inf'), 'mae': float('inf'), 'mape': float('inf'),
                        'rmse_std': 0, 'r2_std': 0, 'mae_std': 0, 'mape_std': 0}
        }
    else:
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
        'optimal_state_features': sorted(list(set(optimal_state_features))),
        'optimal_expl_features': sorted(list(set(optimal_expl_features))),
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
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        json.dump(optimization_result, f, ensure_ascii=False, indent=2, default=convert_numpy)

    print(f"\n优化结果已保存: {result_file}")

    return optimization_result


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
    print(f"  状态变量 (s): {state_features}")
    print(f"  解释变量 (x): {expl_features}")
    print(f"  隶属函数数量 (M): {M}")

    # 准备数据
    state_indices = [feature_names.index(f) for f in state_features]
    expl_indices = [feature_names.index(f) for f in expl_features]

    X_s = X_scaled_tensor[:, state_indices]
    X_x = X_scaled_tensor[:, expl_indices]
    y = y_scaled_tensor

    # 划分训练测试集
    # <--- 修改: 1. 保存 test_indices 以便后续SHAP使用
    train_indices, test_indices = train_test_split(
        range(len(X_s)), test_size=0.2, random_state=42
    )
    # ---> 修改结束

    X_s_train, X_x_train, y_train = X_s[train_indices], X_x[train_indices], y[train_indices]
    X_s_test, X_x_test, y_test = X_s[test_indices], X_x[test_indices], y[test_indices]

    # 创建和训练最终模型
    n_state_features = len(state_features)
    membfuncs_config = create_membfuncs_config(n_state_features, M)

    final_model = SANFIS(
        membfuncs=membfuncs_config,
        n_input=n_state_features,
        to_device='cpu',
        scale='Std'
    ).to(device)

    optimizer = torch.optim.RMSprop(final_model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss(reduction='mean')

    print(f"\n开始训练最终模型 (3000轮)...")

    # ... (训练循环部分保持不变) ...
    final_model.train()
    for epoch in range(3000):
        optimizer.zero_grad()
        train_pred = final_model(X_s_train, X_x_train)
        train_loss = loss_function(train_pred, y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch % 200 == 0:
            final_model.eval()
            with torch.no_grad():
                val_pred = final_model(X_s_test, X_x_test)
                val_loss = loss_function(val_pred, y_test)
                print(f"  Epoch {epoch}: Train Loss = {train_loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
            final_model.train()

    # 最终评估
    final_model.eval()
    with torch.no_grad():
        y_pred_scaled = final_model(X_s_test, X_x_test)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().detach().numpy())
    y_test_original = scaler_y.inverse_transform(y_test.cpu().detach().numpy())

    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)

    print(f"\n最终模型性能:")
    print(f"  MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")

    model_save_path = os.path.join(save_dir, f'final_bsanfis_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'state_features': state_features, 'expl_features': expl_features,
        'membfuncs_config': membfuncs_config,
        'optimization_result': optimization_result,
        'performance': {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    }, model_save_path)
    print(f"\n最终BSANFIS模型已保存: {model_save_path}")

    # <--- 修改: 2. 返回更多用于SHAP分析的信息
    return {
        'model': final_model,
        'performance': {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2},
        'state_features': state_features,
        'expl_features': expl_features,
        'test_indices': test_indices,  # 返回测试集索引
        'save_path': model_save_path
    }


# <--- 修改: 3. 新增SHAP分析函数
def perform_shap_analysis(final_model_result, save_dir):
    """
    对最终模型进行SHAP分析并可视化。
    此版本修复了因'both'角色导致的数据重复问题。
    """
    print(f"\n{'=' * 80}")
    print(f"第三阶段：SHAP分析")
    print(f"{'=' * 80}")

    model = final_model_result['model']
    state_features = final_model_result['state_features']
    expl_features = final_model_result['expl_features']
    test_indices = final_model_result['test_indices']

    # <--- 修改: 3. 核心修复逻辑
    # 步骤1: 确定模型使用的所有唯一特征，并保持固定顺序
    all_model_features = sorted(list(set(state_features + expl_features)))

    # 步骤2: 从原始完整数据中，根据唯一特征列表提取数据
    model_feature_indices = [feature_names.index(f) for f in all_model_features]
    X_model_data_full = X_scaled_tensor[:, model_feature_indices]

    # 步骤3: 准备SHAP的背景数据和测试数据
    X_model_train_data = np.delete(X_model_data_full.cpu().numpy(), test_indices, axis=0)
    X_model_test_data = X_model_data_full[test_indices].cpu().numpy()

    # 步骤4: 创建一个能正确处理输入数据的SHAP包装器
    # 包装器需要知道如何从 all_model_features 重建出 state_features 和 expl_features
    s_indices_in_combined = [all_model_features.index(f) for f in state_features]
    x_indices_in_combined = [all_model_features.index(f) for f in expl_features]

    def model_shap_wrapper(X_combined_numpy):
        X_combined_tensor = torch.tensor(X_combined_numpy, dtype=torch.float32).to(device)

        # 根据预先计算好的索引，从组合输入中切分出 X_s 和 X_x
        X_s_shap = X_combined_tensor[:, s_indices_in_combined]
        X_x_shap = X_combined_tensor[:, x_indices_in_combined]

        model.eval()
        with torch.no_grad():
            predictions = model(X_s_shap, X_x_shap)
        return predictions.cpu().numpy()

    # 步骤5: 将测试数据转换为带列名的DataFrame
    X_shap_df = pd.DataFrame(X_model_test_data, columns=all_model_features)

    # 使用一个小子集作为背景数据来加速计算
    background_data = shap.sample(X_model_train_data, 100)
    explainer = shap.KernelExplainer(model_shap_wrapper, background_data)

    print("正在计算SHAP值... (这可能需要一些时间)")
    shap_values = explainer.shap_values(X_shap_df)

    if shap_values.ndim == 3:
        print(f"原始SHAP值维度: {shap_values.shape}。正在压缩为2维...")
        shap_values = np.squeeze(shap_values, axis=-1)
        print(f"压缩后SHAP值维度: {shap_values.shape}")

    # 保存SHAP值
    shap_df = pd.DataFrame(shap_values, columns=X_shap_df.columns)
    shap_csv_path = os.path.join(save_dir, f'shap_values_{timestamp}.csv')
    shap_df.to_csv(shap_csv_path, index=False, encoding='utf-8-sig')
    print(f"SHAP值已保存到: {shap_csv_path}")

    # --- 可视化 ---
    # 1. SHAP摘要图 (Summary Plot)
    print("生成SHAP摘要图...")
    plt.figure()
    shap.summary_plot(shap_values, X_shap_df, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    summary_plot_path = os.path.join(save_dir, f'shap_summary_plot_{timestamp}.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP摘要图已保存到: {summary_plot_path}")

    # 2. SHAP依赖图 (Dependence Plots)
    print("生成SHAP依赖图...")
    for feature in X_shap_df.columns:
        plt.figure()
        shap.dependence_plot(feature, shap_values, X_shap_df, interaction_index="auto", show=False)
        plt.title(f'SHAP Dependence Plot for {feature}')
        plt.tight_layout()
        dependence_plot_path = os.path.join(save_dir, f'shap_dependence_plot_{feature}_{timestamp}.png')
        plt.savefig(dependence_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"所有特征的SHAP依赖图已保存到: {save_dir}")


# ---> 修改结束


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

    # 生成关键分析图表
    print(f"\n生成关键分析图表...")
    plot_key_analysis(optimization_history, save_dir)

    # 进行特征重要性分析
    print(f"\n进行特征重要性分析...")
    feature_importance_df = analyze_feature_importance(optimization_history, save_dir)

    # 第二阶段：使用最优配置训练最终模型
    print(f"\n第二阶段：使用最优配置训练最终模型")
    final_model_result = train_final_bsanfis_model(result_analysis)

    # <--- 修改: 4. 调用更新后的SHAP分析函数
    # 第三阶段：对最终模型进行SHAP分析
    perform_shap_analysis(final_model_result, save_dir)
    # ---> 修改结束

    print(f"\n{'=' * 80}")
    print(f"BSANFIS实验完成！")
    print(f"{'=' * 80}")
    print(f"贝叶斯优化找到的最优配置:")
    print(f"  状态变量: {result_analysis['optimal_state_features']}")
    print(f"  解释变量: {result_analysis['optimal_expl_features']}")
    print(f"  隶属函数数量: {result_analysis['optimal_M']}")
    print(f"  模糊规则总数: {result_analysis['total_rules']}")
    print(f"\n优化过程中的最佳指标 (交叉验证):")
    print(f"  RMSE: {result_analysis['best_detailed_metrics']['rmse']:.6f}")
    print(f"  R²: {result_analysis['best_detailed_metrics']['r2']:.6f}")
    print(f"  MAE: {result_analysis['best_detailed_metrics']['mae']:.6f}")
    print(f"\n最终模型性能 (在测试集上):")
    print(f"  MSE: {final_model_result['performance']['mse']:.6f}")
    print(f"  RMSE: {final_model_result['performance']['rmse']:.6f}")
    print(f"  R²: {final_model_result['performance']['r2']:.6f}")
    print(f"  MAE: {final_model_result['performance']['mae']:.6f}")
    print(f"\n所有结果已保存到: {save_dir}")
