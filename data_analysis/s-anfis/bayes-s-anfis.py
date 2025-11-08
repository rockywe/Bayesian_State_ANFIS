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
roles = ['none', 'state', 'explanatory', 'both']

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
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 减少到3折以加速
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

            # 5. 构建隶属函数配置
            membfuncs_config = []
            for i in range(Ns):
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

            # 6. 创建和训练SANFIS模型
            # 这里需要您的SANFIS实现
            # model = SANFIS(
            #     membfuncs=membfuncs_config,
            #     n_input=Ns,  # 状态变量数量
            #     to_device='cpu',
            #     scale='Std'
            # )

            # 训练模型
            # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
            # loss_function = torch.nn.MSELoss()

            # 简化训练过程 (您可以根据需要调整epochs)
            # epochs = 200  # 减少训练轮数以加速优化
            # model.train()
            # for epoch in range(epochs):
            #     optimizer.zero_grad()
            #     pred = model(X_s_train, X_x_train)
            #     loss = loss_function(pred, y_train)
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #     optimizer.step()

            # 验证
            # model.eval()
            # with torch.no_grad():
            #     y_pred = model(X_s_val, X_x_val)
            #     rmse = torch.sqrt(torch.mean((y_pred - y_val) ** 2)).item()
            #     rmse_scores.append(rmse)

            # ================================
            # 临时替代方案：使用简单模型模拟
            # 您需要用真实的SANFIS代码替换这部分
            # ================================
            from sklearn.ensemble import RandomForestRegressor

            # 合并所有选择的特征
            all_selected_features = sorted(list(set(state_features + explanatory_features)))
            all_indices = [feature_names.index(f) for f in all_selected_features]

            X_train_combined = X_scaled[train_index][:, all_indices]
            X_val_combined = X_scaled[val_index][:, all_indices]
            y_train_np = y_scaled[train_index].flatten()
            y_val_np = y_scaled[val_index].flatten()

            # 使用随机森林作为替代模型
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train_combined, y_train_np)
            y_pred_np = model.predict(X_val_combined)

            rmse = np.sqrt(mean_squared_error(y_val_np, y_pred_np))
            rmse_scores.append(rmse)

            print(f"    第{fold}折 RMSE: {rmse:.6f}")

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
    print(f"评估次数: 50次 (可根据需要调整)")

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

    # 分析特征重要性
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
        'timestamp': timestamp
    }

    # 保存到文件
    result_file = os.path.join(save_dir, f'bsanfis_optimization_result_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(optimization_result, f, ensure_ascii=False, indent=2)

    print(f"\n优化结果已保存: {result_file}")

    return optimization_result


def plot_optimization_history(result, save_dir):
    """绘制优化历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 目标函数值变化
    ax1.plot(result.func_vals, 'b-', linewidth=2)
    ax1.set_title('贝叶斯优化收敛历史', fontsize=14, fontweight='bold')
    ax1.set_xlabel('评估次数')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)

    # 最佳值变化
    best_so_far = np.minimum.accumulate(result.func_vals)
    ax2.plot(best_so_far, 'r-', linewidth=2)
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

    # 运行贝叶斯优化
    optimization_result = run_bayesian_optimization()

    # 分析结果
    result_analysis = analyze_optimization_result(optimization_result)

    # 绘制优化历史
    plot_optimization_history(optimization_result, save_dir)

    print(f"\n{'=' * 80}")
    print(f"BSANFIS贝叶斯优化完成！")
    print(f"{'=' * 80}")
    print(f"找到的最优配置:")
    print(f"  状态变量: {result_analysis['optimal_state_features']}")
    print(f"  解释变量: {result_analysis['optimal_expl_features']}")
    print(f"  隶属函数数量: {result_analysis['optimal_M']}")
    print(f"  最佳RMSE: {result_analysis['best_rmse']:.6f}")
    print(f"  模糊规则总数: {result_analysis['total_rules']}")
    print(f"\n下一步: 使用最优配置训练最终的BSANFIS模型")
