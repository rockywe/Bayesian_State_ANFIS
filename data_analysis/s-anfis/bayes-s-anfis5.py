import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# 导入原始的、未修改的 sanfis 库
try:
    from sanfis import SANFIS
except ImportError:
    print("错误：'sanfis' 库未找到。请通过 'pip install sanfis' 命令安装。")
    exit()

# ================================
# 全局设备配置
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 设置中文字体
import matplotlib as mpl

try:
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
except Exception:
    print("警告：未能设置中文字体，图形中的中文可能无法正常显示。")

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['path.simplify'] = True
plt.rcParams['path.snap'] = True

# 创建保存目录
save_dir = 'sanfis_results/bfs数据'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 加载数据
file_path = './data/脱硫数据整理2.xlsx'
try:
    data_df = pd.read_excel(file_path, sheet_name='Sheet1')
    print(f"成功从 '{file_path}' 加载数据。")
except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径：{file_path}")
    exit()

# 数据预处理
column_rename_dict = {
    '煤气进口流量': 'Gas_Inlet_Flow', '进口煤气温度': 'Gas_Inlet_Temperature', '进口煤气压力': 'Gas_Inlet_Pressure',
    '脱硫液流量': 'Desulfurization_Liquid_Flow', '脱硫液温度': 'Desulfurization_Liquid_Temperature',
    '脱硫液压力': 'Desulfurization_Liquid_Pressure',
    '转速': 'Rotation_Speed', '进口H2S浓度': 'H2S_Inlet_Concentration', '出口H2S浓度': 'H2S_Outlet_Concentration'
}
data_df.rename(columns=column_rename_dict, inplace=True)

feature_names = [
    'Gas_Inlet_Flow', 'Gas_Inlet_Temperature', 'Gas_Inlet_Pressure',
    'Desulfurization_Liquid_Flow', 'Desulfurization_Liquid_Temperature',
    'Desulfurization_Liquid_Pressure', 'Rotation_Speed', 'H2S_Inlet_Concentration'
]
output_feature = 'H2S_Outlet_Concentration'

data_clean = data_df[feature_names + [output_feature]].dropna()
X_full = data_clean[feature_names].values
y_full = data_clean[output_feature].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_full)
y_scaled = scaler_y.fit_transform(y_full)

X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_scaled_tensor = torch.tensor(y_scaled, dtype=torch.float32)

print("\n为贝叶斯优化过程创建固定的训练集和验证集 (80/20)...")
opt_train_indices, opt_val_indices = train_test_split(
    range(len(X_scaled_tensor)), test_size=0.2, random_state=42
)

# ================================
# 贝叶斯优化搜索空间定义
# ================================
roles = ['state', 'explanatory', 'both', 'none']
space = []
for feature in feature_names:
    safe_feature_name = feature.replace('_', '').replace('2', 'two')
    if feature == 'Rotation_Speed':
        space.append(Categorical(['both'], name=f'role_{safe_feature_name}'))
    else:
        space.append(Categorical(roles, name=f'role_{safe_feature_name}'))
space.append(Integer(2, 3, name='M'))
print(f"\n贝叶斯优化搜索空间维度: {len(space)}")


# ================================
# SANFIS训练辅助函数
# ================================
def create_membfuncs_config(n_state_features, M):
    membfuncs_config = []
    for i in range(n_state_features):
        mu_values = np.linspace(0.1, 0.9, M).tolist()
        sigma_values = [0.2] * M
        membfuncs_config.append({
            'function': 'gaussian', 'n_memb': M,
            'params': {
                'mu': {'value': mu_values, 'trainable': True},
                'sigma': {'value': sigma_values, 'trainable': True}
            }
        })
    return membfuncs_config


def train_sanfis_model(X_s_train, X_x_train, y_train, X_s_val, X_x_val, y_val, M, epochs=2000):
    try:
        n_state_features = X_s_train.shape[1]
        membfuncs_config = create_membfuncs_config(n_state_features, M)
        model = SANFIS(
            membfuncs=membfuncs_config, n_input=n_state_features, to_device='cpu', scale='Std'
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
            y_val_np, val_pred_np = y_val.cpu().detach().numpy(), val_pred.cpu().detach().numpy()
            mse = mean_squared_error(y_val_np, val_pred_np)
            rmse, r2 = np.sqrt(mse), r2_score(y_val_np, val_pred_np)
            mae = mean_absolute_error(y_val_np, val_pred_np)
            mape = np.mean(np.abs((y_val_np - val_pred_np) / (y_val_np + 1e-8))) * 100
            metrics = {'rmse': rmse, 'r2': r2, 'mse': mse, 'mae': mae, 'mape': mape}
        return metrics
    except Exception as e:
        print(f"      SANFIS训练错误: {e}")
        return {'rmse': 1e10, 'r2': -1e10, 'mse': 1e10, 'mae': 1e10, 'mape': 1e10}


# ================================
# 目标函数
# ================================
optimization_history = []


@use_named_args(space)
def objective(**params):
    try:
        state_features, explanatory_features = [], []
        for feature in feature_names:
            safe_feature_name = feature.replace('_', '').replace('2', 'two')
            role = params[f'role_{safe_feature_name}']
            if role in ['state', 'both']: state_features.append(feature)
            if role in ['explanatory', 'both']: explanatory_features.append(feature)
        M = params['M']
        state_features, explanatory_features = sorted(list(set(state_features))), sorted(
            list(set(explanatory_features)))
        current_eval_count = len(optimization_history) + 1
        print(f"\n--- 评估组合 (历史记录 {current_eval_count}) ---")
        print(f"状态(s): {state_features}\n解释(x): {explanatory_features}\nM: {M}")
        if len(state_features) != len(explanatory_features):
            print(">> 跳过: s和x变量数量不匹配。返回高惩罚值。")
            return 1e10
        if not state_features or not explanatory_features:
            print(">> 跳过: 变量列表为空。返回高惩罚值。")
            return 1e10
        if len(state_features) > 6:
            print(f">> 跳过: 状态变量过多({len(state_features)} > 6)。返回高惩罚值。")
            return 1e10
        Ns, num_rules = len(state_features), M ** len(state_features)
        print(f"规则数量: {num_rules}")
        state_indices = [feature_names.index(f) for f in state_features]
        expl_indices = [feature_names.index(f) for f in explanatory_features]
        X_s, X_x, y = X_scaled_tensor[:, state_indices], X_scaled_tensor[:, expl_indices], y_scaled_tensor
        X_s_train, X_x_train, y_train = X_s[opt_train_indices], X_x[opt_train_indices], y[opt_train_indices]
        X_s_val, X_x_val, y_val = X_s[opt_val_indices], X_x[opt_val_indices], y[opt_val_indices]
        print(f"  正在训练模型 (2000 epochs)...")
        metrics = train_sanfis_model(X_s_train, X_x_train, y_train, X_s_val, X_x_val, y_val, M)
        if np.isnan(metrics['rmse']): metrics['rmse'] = 1e10
        print(f"  验证集指标 - RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")
        guidance_factor = max(1.1 - metrics['r2'], 0.01)
        final_score = metrics['rmse'] * guidance_factor
        print(f"  最终优化分数: {final_score:.6f}")
        evaluation_record = {
            'evaluation_id': current_eval_count, 'state_features': state_features,
            'explanatory_features': explanatory_features, 'M': M, 'num_rules': num_rules,
            'metrics': metrics, 'final_score': final_score, 'raw_params': params
        }
        optimization_history.append(evaluation_record)
        print(f"--- 有效评估完成，当前有效计数: {len(optimization_history)} ---")
        return final_score
    except Exception as e:
        print(f"在objective函数顶层捕获到严重错误: {e}")
        traceback.print_exc()
        return 1e12


# ================================
# 运行贝叶斯优化的函数
# ================================
def run_bayesian_optimization_iteratively(target_evals=100, batch_size=25, max_total_calls=500):
    print(f"\n{'=' * 80}")
    print(f"开始迭代式BSANFIS贝叶斯优化")
    print(f"目标有效评估次数 (贝叶斯优化轮次): {target_evals}")
    print(f"每个模型的训练轮次 (Epochs): 2000")
    print(f"每批次评估上限: {batch_size}")
    print(f"{'=' * 80}")
    x_iters_total, y_iters_total = [], []
    total_calls_made, batch_num = 0, 1
    while len(optimization_history) < target_evals:
        if total_calls_made >= max_total_calls:
            print(f"\n警告: 总调用次数已达上限 ({max_total_calls})，但仍未找到足够的有效配置。")
            print(f"当前找到 {len(optimization_history)}/{target_evals} 个。提前终止优化。")
            break
        print(f"\n--- 第 {batch_num} 批次优化开始 ---")
        print(f"当前进度: {len(optimization_history)} / {target_evals} 个有效评估")

        # **↓↓↓ 错误修复点 ↓↓↓**
        # 明确检查列表/数组的长度，而不是它的“真假值”
        if len(x_iters_total) > 0:
            x0_arg = x_iters_total
            y0_arg = y_iters_total
            initial_points = 0
        else:
            x0_arg = None
            y0_arg = None
            initial_points = 10  # 只在第一批次进行随机探索
        # **↑↑↑ 错误修复点 ↑↑↑**

        result = gp_minimize(
            func=objective, dimensions=space, n_calls=batch_size, n_initial_points=initial_points,
            random_state=42 + batch_num, n_jobs=-1, acq_func='EI', x0=x0_arg, y0=y0_arg
        )

        x_iters_total, y_iters_total = result.x_iters, result.func_vals
        total_calls_made += batch_size
        batch_num += 1
        print(f"--- 第 {batch_num - 1} 批次优化结束 ---")
        print(f"本批次后进度: {len(optimization_history)} / {target_evals} 个有效评估")
        print(f"累计总调用次数: {total_calls_made}")
    print(f"\n{'=' * 80}\n优化流程结束。最终找到 {len(optimization_history)} 个有效配置。\n{'=' * 80}")
    return result


# ================================
# 可视化和分析函数
# ================================
def plot_optimization_progress(optimization_history, save_dir, timestamp):
    if not optimization_history: print("优化历史为空，无法绘制进度图。"); return
    iterations = np.arange(1, len(optimization_history) + 1)
    r2_values = [r['metrics']['r2'] for r in optimization_history]
    rmse_values = [r['metrics']['rmse'] for r in optimization_history]
    successful_indices = [i for i, r2 in enumerate(r2_values) if -10 < r2 < 10]
    iter_plot, r2_plot, rmse_plot = [iterations[i] for i in successful_indices], [r2_values[i] for i in
                                                                                  successful_indices], [rmse_values[i]
                                                                                                        for i in
                                                                                                        successful_indices]
    r2_for_best, rmse_for_best = [v if -10 < v < 10 else -np.inf for v in r2_values], [v if v < 1e9 else np.inf for v in
                                                                                       rmse_values]
    best_r2_so_far, best_rmse_so_far = [max(r2_for_best[:i + 1]) for i in range(len(r2_for_best))], [
        min(rmse_for_best[:i + 1]) for i in range(len(rmse_for_best))]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True);
    sns.set_style("whitegrid")
    ax1.scatter(iter_plot, r2_plot, c='blue', label='当次迭代 R²', alpha=0.7, s=30, zorder=2);
    ax1.step(iterations, best_r2_so_far, where='post', c='red', label='迄今最优 R²', linewidth=2.5, zorder=3);
    ax1.set_ylabel('$R^2$ 迭代值', fontsize=14);
    ax1.set_title('R² 优化过程', fontsize=16, fontweight='bold');
    ax1.legend(loc='lower right');
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5);
    ax1.set_ylim(-1.1, 1.1)
    ax2.scatter(iter_plot, rmse_plot, c='blue', label='当次迭代 RMSE', alpha=0.7, s=30, zorder=2);
    ax2.step(iterations, best_rmse_so_far, where='post', c='red', label='迄今最优 RMSE', linewidth=2.5, zorder=3);
    ax2.set_ylabel('RMSE 迭代值 (对数坐标)', fontsize=14);
    ax2.set_xlabel('有效评估次数', fontsize=14);
    ax2.set_title('RMSE 优化过程', fontsize=16, fontweight='bold');
    ax2.legend(loc='upper right');
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5);
    ax2.set_yscale('log')
    plt.tight_layout(pad=2.0);
    save_path = os.path.join(save_dir, f'optimization_progress_{timestamp}.png');
    plt.savefig(save_path, dpi=300, bbox_inches='tight');
    plt.show();
    print(f"优化过程图已保存到: {save_path}")


def analyze_optimization_result(result):
    print(f"\n{'=' * 80}\n贝叶斯优化结果分析\n{'=' * 80}")
    if not optimization_history: print("警告: 优化历史为空，无法找到最佳记录。"); return None
    best_record = min(optimization_history, key=lambda x: x['final_score'])
    optimal_state_features, optimal_expl_features, optimal_M = best_record['state_features'], best_record[
        'explanatory_features'], best_record['M']
    print(
        f"\n最优配置:\n  状态(s): {optimal_state_features}\n  解释(x): {optimal_expl_features}\n  M: {optimal_M}\n  规则数: {optimal_M ** len(optimal_state_features)}")
    if 'metrics' in best_record and best_record['metrics']: print(
        f"\n最优配置性能 (验证集):\n  RMSE: {best_record['metrics']['rmse']:.6f}, R²: {best_record['metrics']['r2']:.6f}")
    best_params_from_skopt = dict(zip([s.name for s in space], result.x))
    optimization_result_data = {
        'best_objective_value': result.fun, 'best_params_from_skopt': best_params_from_skopt,
        'best_config_from_history': {
            'state_features': optimal_state_features, 'expl_features': optimal_expl_features, 'M': optimal_M,
            'total_rules': optimal_M ** len(optimal_state_features), 'metrics': best_record.get('metrics', {}),
            'final_score': best_record.get('final_score')
        },
        'optimization_history': optimization_history, 'timestamp': timestamp, 'feature_names': feature_names
    }
    result_file = os.path.join(save_dir, f'bsanfis_optimization_result_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, (np.bool_, bool)): return bool(obj)
            if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert_numpy(i) for i in obj]
            return obj

        json.dump(optimization_result_data, f, ensure_ascii=False, indent=2, default=convert_numpy)
    print(f"\n优化结果已保存: {result_file}")
    return optimization_result_data


def train_final_bsanfis_model(optimization_result):
    print(f"\n{'=' * 80}\n使用最优配置训练最终BSANFIS模型\n{'=' * 80}")
    best_config = optimization_result['best_config_from_history']
    state_features, expl_features, M = best_config['state_features'], best_config['expl_features'], best_config['M']
    state_indices, expl_indices = [feature_names.index(f) for f in state_features], [feature_names.index(f) for f in
                                                                                     expl_features]
    X_s, X_x, y = X_scaled_tensor[:, state_indices], X_scaled_tensor[:, expl_indices], y_scaled_tensor
    final_train_indices, final_test_indices = train_test_split(range(len(X_s)), test_size=0.2, random_state=123)
    X_s_train, X_x_train, y_train = X_s[final_train_indices], X_x[final_train_indices], y[final_train_indices]
    X_s_test, X_x_test, y_test = X_s[final_test_indices], X_x[final_test_indices], y[final_test_indices]
    final_model = SANFIS(
        membfuncs=create_membfuncs_config(len(state_features), M), n_input=len(state_features), to_device='cpu',
        scale='Std'
    ).to(device)
    optimizer, loss_function = torch.optim.Adam(final_model.parameters(), lr=0.001), torch.nn.MSELoss()
    X_s_train, X_x_train, y_train = X_s_train.to(device), X_x_train.to(device), y_train.to(device)
    X_s_test, X_x_test, y_test = X_s_test.to(device), X_x_test.to(device), y_test.to(device)
    final_epochs = 2000
    print(f"\n开始训练最终模型 ({final_epochs}轮)...")
    final_model.train()
    for epoch in range(final_epochs):
        optimizer.zero_grad()
        train_pred = final_model(X_s_train, X_x_train)
        train_loss = loss_function(train_pred, y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()
        if (epoch + 1) % 200 == 0 or epoch == 0:
            final_model.eval()
            with torch.no_grad():
                val_pred = final_model(X_s_test, X_x_test)
                print(
                    f"  Epoch {epoch + 1}/{final_epochs}: Train Loss = {train_loss.item():.6f}, Test Loss = {loss_function(val_pred, y_test).item():.6f}")
            final_model.train()
    final_model.eval()
    with torch.no_grad():
        y_pred_scaled = final_model(X_s_test, X_x_test)
    y_pred, y_test_original = scaler_y.inverse_transform(
        y_pred_scaled.cpu().detach().numpy()), scaler_y.inverse_transform(y_test.cpu().detach().numpy())
    mse, rmse, r2, mae = mean_squared_error(y_test_original, y_pred), np.sqrt(
        mean_squared_error(y_test_original, y_pred)), r2_score(y_test_original, y_pred), mean_absolute_error(
        y_test_original, y_pred)
    print(f"\n最终模型在独立测试集上的性能:\n  MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
    model_save_path = os.path.join(save_dir, f'final_bsanfis_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': final_model.state_dict(), 'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'state_features': state_features, 'expl_features': expl_features,
        'performance': {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    }, model_save_path)
    print(f"\n最终BSANFIS模型已保存: {model_save_path}")
    return {'model': final_model, 'performance': {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}}


# ================================
# 主实验流程
# ================================
if __name__ == "__main__":
    print(f"开始BSANFIS实验 - 时间戳: {timestamp}")
    result_obj = run_bayesian_optimization_iteratively(target_evals=100, batch_size=25)
    if optimization_history:
        result_analysis = analyze_optimization_result(result_obj)
        plot_optimization_progress(optimization_history, save_dir, timestamp)
        final_model_result = train_final_bsanfis_model(result_analysis)
    else:
        print("\n优化过程中未找到任何有效配置，无法进行分析、绘图和最终模型训练。")
    print(f"\n{'=' * 80}\nBSANFIS实验完成！\n所有结果已保存到: {save_dir}\n{'=' * 80}")
