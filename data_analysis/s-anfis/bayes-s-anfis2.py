import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
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


def train_sanfis_model(X_s_train, X_x_train, y_train, X_s_val, X_x_val, y_val, M, epochs=200):
    """
    训练SANFIS模型

    参数:
    - X_s_train, X_s_val: 状态变量数据
    - X_x_train, X_x_val: 解释变量数据
    - y_train, y_val: 目标变量
    - M: 隶属函数数量
    - epochs: 训练轮数

    返回:
    - validation_rmse: 验证集RMSE
    """
    try:
        n_state_features = X_s_train.shape[1]
        n_expl_features = X_x_train.shape[1]

        # 创建隶属函数配置
        membfuncs_config = create_membfuncs_config(n_state_features, M)

        # 初始化SANFIS模型
        model = SANFIS(
            membfuncs=membfuncs_config,
            n_input=n_state_features,  # 状态变量数量
            to_device='cpu',
            scale='Std'
        )

        # 设置优化器
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        loss_function = torch.nn.MSELoss(reduction='mean')

        # 训练数据准备
        train_data = [X_s_train, X_x_train, y_train]

        # 训练模型
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # SANFIS前向传播：model(state_vars, explanatory_vars)
            train_pred = model(X_s_train, X_x_train)
            train_loss = loss_function(train_pred, y_train)

            train_loss.backward()

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 每50轮检查一次验证损失
            if (epoch + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_s_val, X_x_val)
                    val_loss = loss_function(val_pred, y_val)
                model.train()

        # 最终验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_s_val, X_x_val)
            val_rmse = torch.sqrt(torch.mean((val_pred - y_val) ** 2)).item()

        return val_rmse

    except Exception as e:
        print(f"      SANFIS训练错误: {e}")
        return 1e10  # 返回一个很大的误差值作为惩罚


# ================================
# 目标函数定义
# ================================

@use_named_args(space)
def objective(**params):
    """
    BSANFIS贝叶斯优化目标函数
    输入: 8个特征的角色 + 隶属函数数量M
    输出: 交叉验证的平均RMSE (待最小化)
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
        # role == 'none' 时什么都不做

    M = params['M']

    # 去重并排序
    state_features = sorted(list(set(state_features)))
    explanatory_features = sorted(list(set(explanatory_features)))

    print(f"\n{'=' * 60}")
    print(f"测试参数组合:")
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
    MAX_RULES = 200  # 设置规则数量上限

    if num_rules > MAX_RULES:
        print(f"跳过: 规则数量过多 ({num_rules} > {MAX_RULES})")
        return 1e10

    print(f"规则数量: {num_rules}")

    # 3. 准备数据
    try:
        # 获取特征索引
        state_indices = [feature_names.index(f) for f in state_features]
        expl_indices = [feature_names.index(f) for f in explanatory_features]

        X_s = X_scaled_tensor[:, state_indices]
        X_x = X_scaled_tensor[:, expl_indices]
        y = y_scaled_tensor

        print(f"数据形状: S={X_s.shape}, X={X_x.shape}, y={y.shape}")

    except Exception as e:
        print(f"数据准备错误: {e}")
        return 1e10

    # 4. K折交叉验证
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 使用3折交叉验证
    rmse_scores = []

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

            # 训练SANFIS模型
            val_rmse = train_sanfis_model(
                X_s_train, X_x_train, y_train,
                X_s_val, X_x_val, y_val,
                M, epochs=200  # 可以根据需要调整训练轮数
            )

            rmse_scores.append(val_rmse)
            print(f"    第{fold}折 RMSE: {val_rmse:.6f}")

        except Exception as e:
            print(f"    第{fold}折训练失败: {e}")
            rmse_scores.append(1e10)  # 惩罚失败的配置

    # 计算平均RMSE
    avg_rmse = np.mean(rmse_scores)
    print(f"平均 RMSE: {avg_rmse:.6f}")
    print(f"{'=' * 60}")

    return avg_rmse


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


def analyze_optimization_result(result):
    """分析优化结果"""
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

    print(f"\n最优配置:")
    print(f"状态变量 (s): {optimal_state_features}")
    print(f"解释变量 (x): {optimal_expl_features}")
    print(f"隶属函数数量 (M): {optimal_M}")
    print(f"规则总数: {optimal_M ** len(optimal_state_features)}")

    # 分析特征角色分配
    print(f"\n特征角色分配:")
    for feature in feature_names:
        safe_feature_name = feature.replace('_', '').replace('2', 'two')
        role = best_params[f'role_{safe_feature_name}']
        print(f"  {feature}: {role}")

    # 保存结果
    optimization_result = {
        'best_rmse': result.fun,
        'best_params': best_params,
        'optimal_state_features': optimal_state_features,
        'optimal_expl_features': optimal_expl_features,
        'optimal_M': optimal_M,
        'total_rules': optimal_M ** len(optimal_state_features),
        'optimization_history': {
            'func_vals': result.func_vals,
            'x_iters': result.x_iters
        },
        'timestamp': timestamp,
        'feature_names': feature_names
    }

    # 保存到文件
    result_file = os.path.join(save_dir, f'bsanfis_optimization_result_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(optimization_result, f, ensure_ascii=False, indent=2)

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


def plot_optimization_history(result, save_dir):
    """绘制优化历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 目标函数值变化
    ax1.plot(result.func_vals, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title('贝叶斯优化收敛历史', fontsize=14, fontweight='bold')
    ax1.set_xlabel('评估次数')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)

    # 最佳值变化
    best_so_far = np.minimum.accumulate(result.func_vals)
    ax2.plot(best_so_far, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_title('最佳RMSE变化', fontsize=14, fontweight='bold')
    ax2.set_xlabel('评估次数')
    ax2.set_ylabel('最佳RMSE')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'optimization_history_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


# ================================
# 主实验流程
# ================================

if __name__ == "__main__":
    print(f"开始BSANFIS实验 - 时间戳: {timestamp}")

    # 第一阶段：贝叶斯优化寻找最优结构
    print(f"\n第一阶段：贝叶斯优化寻找最优结构")
    optimization_result = run_bayesian_optimization()

    # 分析优化结果
    result_analysis = analyze_optimization_result(optimization_result)

    # 绘制优化历史
    plot_optimization_history(optimization_result, save_dir)

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
    print(f"  优化过程最佳RMSE: {result_analysis['best_rmse']:.6f}")
    print(f"  模糊规则总数: {result_analysis['total_rules']}")
    print(f"\n最终模型性能:")
    print(f"  MSE: {final_model_result['performance']['mse']:.6f}")
    print(f"  RMSE: {final_model_result['performance']['rmse']:.6f}")
    print(f"  R²: {final_model_result['performance']['r2']:.6f}")
    print(f"\n模型保存位置: {final_model_result['save_path']}")
