import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sanfis import SANFIS

# ================================
# 全局设备配置
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
save_dir = 'sanfis_results/comparison_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ================================
# 数据加载与预处理
# ================================
try:
    data_df = pd.read_excel('./data/脱硫数据整理2.xlsx', sheet_name='Sheet1')
except FileNotFoundError:
    print("错误：数据文件未找到，请检查路径。")
    exit()

column_rename_dict = {
    '煤气进口流量': 'Gas_Inlet_Flow', '进口煤气温度': 'Gas_Inlet_Temperature',
    '进口煤气压力': 'Gas_Inlet_Pressure', '脱硫液流量': 'Desulfurization_Liquid_Flow',
    '脱硫液温度': 'Desulfurization_Liquid_Temperature', '脱硫液压力': 'Desulfurization_Liquid_Pressure',
    '转速': 'Rotation_Speed', '进口H2S浓度': 'H2S_Inlet_Concentration',
    '出口H2S浓度': 'H2S_Outlet_Concentration'
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
print("数据预处理完成。")

# ================================
# BS-ANFIS 相关函数 (来自您的原始代码)
# ================================
space = [Categorical(['state', 'explanatory', 'both'], name=f'role_{f.replace("_", "")}') for f in feature_names]
space.append(Integer(2, 4, name='M'))
space[feature_names.index('Rotation_Speed')] = Categorical(['both'], name='role_RotationSpeed')


def create_membfuncs_config(n_state_features, M):
    configs = []
    for _ in range(n_state_features):
        configs.append({
            'function': 'gaussian', 'n_memb': M,
            'params': {
                'mu': {'value': np.linspace(0.1, 0.9, M).tolist(), 'trainable': True},
                'sigma': {'value': [0.2] * M, 'trainable': True}
            }
        })
    return configs


def train_sanfis_model_cv(X_s_train, X_x_train, y_train, X_s_val, X_x_val, y_val, M, epochs=1000):
    try:
        n_state_features = X_s_train.shape[1]
        model = SANFIS(membfuncs=create_membfuncs_config(n_state_features, M), n_input=n_state_features).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X_s_train.to(device), X_x_train.to(device))
            loss = loss_fn(pred, y_train.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_s_val.to(device), X_x_val.to(device)).cpu().numpy()
        y_val_np = y_val.numpy()
        return {'rmse': np.sqrt(mean_squared_error(y_val_np, val_pred))}
    except Exception:
        return {'rmse': 1e10}


optimization_history = []


@use_named_args(space)
def objective(**params):
    state_features, expl_features = [], []
    for feature in feature_names:
        role = params[f'role_{feature.replace("_", "")}']
        if role in ['state', 'both']: state_features.append(feature)
        if role in ['explanatory', 'both']: expl_features.append(feature)
    M = params['M']
    state_features, expl_features = sorted(list(set(state_features))), sorted(list(set(expl_features)))
    if not state_features or not expl_features or 'Rotation_Speed' not in state_features or 'Rotation_Speed' not in expl_features:
        return 1e12
    if M ** len(state_features) > 3000: return 1e10

    state_idx = [feature_names.index(f) for f in state_features]
    expl_idx = [feature_names.index(f) for f in expl_features]
    X_s, X_x = X_scaled_tensor[:, state_idx], X_scaled_tensor[:, expl_idx]

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(X_s):
        metrics = train_sanfis_model_cv(X_s[train_idx], X_x[train_idx], y_scaled_tensor[train_idx],
                                        X_s[val_idx], X_x[val_idx], y_scaled_tensor[val_idx], M)
        rmses.append(metrics['rmse'])

    avg_rmse = np.nanmean(rmses)
    optimization_history.append({'params': params, 'avg_rmse': avg_rmse})
    return avg_rmse


def run_bayesian_optimization():
    print("\n--- 运行BS-ANFIS贝叶斯优化 (这可能需要一些时间) ---")
    # 为了快速演示，减少调用次数。在实际研究中建议 n_calls=50 或更高。
    return gp_minimize(func=objective, dimensions=space, n_calls=10, n_initial_points=5, random_state=42, n_jobs=-1)


### 新增部分 ###
# ==================================
# 定义其他对比模型
# ==================================
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1):
        super(ANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1), nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2), nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        return self.network(x)


# ==================================
# 结果可视化函数
# ==================================
def plot_comparison_results(results, save_dir):
    """将所有模型的性能指标进行可视化对比"""
    df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
    fig.suptitle('模型性能对比', fontsize=20, fontweight='bold')

    metrics = ['rmse', 'r2', 'mae']
    titles = ['均方根误差 (RMSE - 越低越好)', 'R²分数 (R-squared - 越高越好)', '平均绝对误差 (MAE - 越低越好)']
    colors = ['#4c72b0', '#55a868', '#c44e52']

    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        sns.barplot(x='Model', y=metric, data=df.sort_values(metric, ascending=(metric != 'r2')), ax=axes[i],
                    palette=[color])
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel('模型', fontsize=12)
        axes[i].set_ylabel('值', fontsize=12)
        axes[i].tick_params(axis='x', rotation=15)

        # 在条形图上添加数值标签
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height():.4f}',
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, 9),
                             textcoords='offset points',
                             fontsize=10)
        if metric == 'r2':
            axes[i].set_ylim(bottom=max(0, df['r2'].min() - 0.1), top=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, f'model_comparison_{timestamp}.png')
    plt.savefig(save_path)
    print(f"\n对比图已保存到: {save_path}")
    plt.show()


# ==================================
# 模型训练与评估主流程
# ==================================
def run_comparison_experiment():
    print(f"\n{'=' * 80}")
    print("开始进行多模型对比实验")
    print(f"{'=' * 80}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    X_train_t, y_train_t = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train,
                                                                                               dtype=torch.float32).to(
        device)
    X_test_t, y_test_t = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test,
                                                                                            dtype=torch.float32).to(
        device)
    y_test_orig = scaler_y.inverse_transform(y_test)

    comparison_results = {}

    # --- 模型1: 线性回归 (LR) ---
    print("\n--- 训练与评估: 1. 线性回归 (LR) ---")
    lr_model = LinearRegression().fit(X_train, y_train)
    y_pred_lr = scaler_y.inverse_transform(lr_model.predict(X_test))
    comparison_results['LR'] = {
        'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_lr)),
        'r2': r2_score(y_test_orig, y_pred_lr),
        'mae': mean_absolute_error(y_test_orig, y_pred_lr)
    }
    print(
        f"LR 性能: RMSE={comparison_results['LR']['rmse']:.4f}, R²={comparison_results['LR']['r2']:.4f}, MAE={comparison_results['LR']['mae']:.4f}")

    # --- 模型2: 人工神经网络 (ANN) ---
    print("\n--- 训练与评估: 2. 人工神经网络 (ANN) ---")
    ann_model = ANN(input_size=X_train.shape[1]).to(device)
    optimizer_ann = torch.optim.RMSprop(ann_model.parameters(), lr=0.001)
    loss_fn_ann = nn.MSELoss()
    for epoch in range(2000):
        ann_model.train()
        optimizer_ann.zero_grad()
        loss = loss_fn_ann(ann_model(X_train_t), y_train_t)
        loss.backward()
        optimizer_ann.step()

    ann_model.eval()
    with torch.no_grad():
        y_pred_ann = scaler_y.inverse_transform(ann_model(X_test_t).cpu().numpy())
    comparison_results['ANN'] = {
        'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_ann)),
        'r2': r2_score(y_test_orig, y_pred_ann),
        'mae': mean_absolute_error(y_test_orig, y_pred_ann)
    }
    print(
        f"ANN 性能: RMSE={comparison_results['ANN']['rmse']:.4f}, R²={comparison_results['ANN']['r2']:.4f}, MAE={comparison_results['ANN']['mae']:.4f}")

    # --- 模型3: 标准 ANFIS ---
    print("\n--- 训练与评估: 3. 标准 ANFIS ---")
    n_features = X_train.shape[1]
    M_anfis = 3  # 固定隶属函数数量
    num_rules = M_anfis ** n_features
    print(f"标准ANFIS配置: M={M_anfis}, 输入特征={n_features}, 规则数={num_rules}")

    if num_rules > 5000:  # 避免规则爆炸
        print("规则数过多，跳过标准ANFIS。")
        comparison_results['ANFIS'] = {'rmse': float('inf'), 'r2': -float('inf'), 'mae': float('inf')}
    else:
        membfuncs_config = create_membfuncs_config(n_features, M_anfis)
        anfis_model = SANFIS(membfuncs=membfuncs_config, n_input=n_features).to(device)
        optimizer_anfis = torch.optim.RMSprop(anfis_model.parameters(), lr=0.001)
        loss_fn_anfis = nn.MSELoss()

        # 标准ANFIS中，所有输入既是状态变量也是解释变量
        X_train_anfis_s, X_train_anfis_x = X_train_t, X_train_t
        X_test_anfis_s, X_test_anfis_x = X_test_t, X_test_t

        for epoch in range(2000):
            anfis_model.train()
            optimizer_anfis.zero_grad()
            loss = loss_fn_anfis(anfis_model(X_train_anfis_s, X_train_anfis_x), y_train_t)
            loss.backward()
            optimizer_anfis.step()

        anfis_model.eval()
        with torch.no_grad():
            y_pred_anfis = scaler_y.inverse_transform(anfis_model(X_test_anfis_s, X_test_anfis_x).cpu().numpy())
        comparison_results['ANFIS'] = {
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_anfis)),
            'r2': r2_score(y_test_orig, y_pred_anfis),
            'mae': mean_absolute_error(y_test_orig, y_pred_anfis)
        }
        print(
            f"ANFIS 性能: RMSE={comparison_results['ANFIS']['rmse']:.4f}, R²={comparison_results['ANFIS']['r2']:.4f}, MAE={comparison_results['ANFIS']['mae']:.4f}")

    # --- 模型4: BS-ANFIS ---
    bo_result = run_bayesian_optimization()

    # 从优化历史中找到最佳参数
    best_config = min(optimization_history, key=lambda x: x['avg_rmse'])['params']

    state_features, expl_features = [], []
    for feature in feature_names:
        role = best_config[f'role_{feature.replace("_", "")}']
        if role in ['state', 'both']: state_features.append(feature)
        if role in ['explanatory', 'both']: expl_features.append(feature)
    M_bsanfis = best_config['M']
    state_features, expl_features = sorted(list(set(state_features))), sorted(list(set(expl_features)))

    print("\n--- 训练与评估: 4. 最终 BS-ANFIS 模型 ---")
    print(f"最优配置: M={M_bsanfis}, 状态变量={state_features}, 解释变量={expl_features}")

    state_idx = [feature_names.index(f) for f in state_features]
    expl_idx = [feature_names.index(f) for f in expl_features]

    X_train_s, X_test_s = X_train_t[:, state_idx], X_test_t[:, state_idx]
    X_train_x, X_test_x = X_train_t[:, expl_idx], X_test_t[:, expl_idx]

    bsanfis_model = SANFIS(membfuncs=create_membfuncs_config(len(state_features), M_bsanfis),
                           n_input=len(state_features)).to(device)
    optimizer_bsanfis = torch.optim.RMSprop(bsanfis_model.parameters(), lr=0.001)
    loss_fn_bsanfis = nn.MSELoss()

    for epoch in range(2000):
        bsanfis_model.train()
        optimizer_bsanfis.zero_grad()
        loss = loss_fn_bsanfis(bsanfis_model(X_train_s, X_train_x), y_train_t)
        loss.backward()
        optimizer_bsanfis.step()

    bsanfis_model.eval()
    with torch.no_grad():
        y_pred_bsanfis = scaler_y.inverse_transform(bsanfis_model(X_test_s, X_test_x).cpu().numpy())
    comparison_results['BS-ANFIS'] = {
        'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_bsanfis)),
        'r2': r2_score(y_test_orig, y_pred_bsanfis),
        'mae': mean_absolute_error(y_test_orig, y_pred_bsanfis)
    }
    print(
        f"BS-ANFIS 性能: RMSE={comparison_results['BS-ANFIS']['rmse']:.4f}, R²={comparison_results['BS-ANFIS']['r2']:.4f}, MAE={comparison_results['BS-ANFIS']['mae']:.4f}")

    # --- 结果汇总与可视化 ---
    print(f"\n{'=' * 80}")
    print("所有模型评估完成，生成最终对比图...")
    print(f"{'=' * 80}")
    plot_comparison_results(comparison_results, save_dir)


if __name__ == "__main__":
    # 运行完整的对比实验
    run_comparison_experiment()
    print("\n所有实验和对比已成功完成！")

