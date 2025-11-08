import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sanfis import SANFIS

# ===== Matplotlib Configuration =====
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica World', 'Arial', 'Arial Unicode MS']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['path.simplify'] = True
plt.rcParams['path.snap'] = True
# ================================
# 全局设备配置
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 创建保存目录
save_dir = 'sanfis_results/fixed_params_comparison'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ================================
# 数据加载与预处理
# ================================
try:
    data_df = pd.read_excel('./data/脱硫数据整理2.xlsx', sheet_name='Sheet1')
except FileNotFoundError:
    print("错误：数据文件 './data/脱硫数据整理2.xlsx' 未找到，请检查路径。")
    exit()

column_rename_dict = {
    '煤气进口流量': 'Gas_Inlet_Flow', '进口煤气温度': 'Gas_Inlet_Temperature',
    '进口煤气压力': 'Gas_Inlet_Pressure', '脱硫液流量': 'Desulfurization_Liquid_Flow',
    '脱硫液温度': 'Desulfurization_Liquid_Temperature', '脱硫液压力': 'Desulfurization_Liquid_Pressure',
    '转速': 'Rotation_Speed', '进口H2S浓度': 'H2S_Inlet_Concentration',
    '出口H2S浓度': 'H2S_Outlet_Concentration'
}
data_df.rename(columns=column_rename_dict, inplace=True)

# 所有的原始输入特征名称
all_feature_names = [
    'Gas_Inlet_Flow', 'Gas_Inlet_Temperature', 'Gas_Inlet_Pressure',
    'Desulfurization_Liquid_Flow', 'Desulfurization_Liquid_Temperature',
    'Desulfurization_Liquid_Pressure', 'Rotation_Speed', 'H2S_Inlet_Concentration'
]
output_feature = 'H2S_Outlet_Concentration'

data_clean = data_df[all_feature_names + [output_feature]].dropna()
X_full = data_clean[all_feature_names].values
y_full = data_clean[output_feature].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_full)
y_scaled = scaler_y.fit_transform(y_full)
X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_scaled_tensor = torch.tensor(y_scaled, dtype=torch.float32)
print("数据预处理完成。")


# ================================
# 辅助函数
# ================================
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


# ================================
# BS-ANFIS (使用预设参数) 训练函数
# ================================
def run_preconfigured_bsanfis_and_get_results(best_params, train_indices, test_indices):
    """
    使用预设的最优参数来配置、训练和评估BS-ANFIS模型。
    """
    print("\n--- 训练与评估: 4. BS-ANFIS (使用预设最优配置) ---")

    # 1. 解析预设参数以确定特征角色
    config_to_df_col_map = {
        'GasInletFlow': 'Gas_Inlet_Flow', 'GasInletTemperature': 'Gas_Inlet_Temperature',
        'GasInletPressure': 'Gas_Inlet_Pressure', 'DesulfurizationLiquidFlow': 'Desulfurization_Liquid_Flow',
        'DesulfurizationLiquidTemperature': 'Desulfurization_Liquid_Temperature',
        'DesulfurizationLiquidPressure': 'Desulfurization_Liquid_Pressure',
        'RotationSpeed': 'Rotation_Speed', 'HtwoSInletConcentration': 'H2S_Inlet_Concentration'
    }

    state_features, expl_features = [], []
    for key, role in best_params.items():
        if key.startswith("role_"):
            config_key = key.replace("role_", "")
            df_col = config_to_df_col_map.get(config_key)
            if df_col:
                if role == 'state':
                    state_features.append(df_col)
                elif role == 'explanatory':
                    expl_features.append(df_col)
                elif role == 'both':
                    state_features.append(df_col)
                    expl_features.append(df_col)

    state_features = sorted(list(set(state_features)))
    expl_features = sorted(list(set(expl_features)))
    M_bsanfis = best_params['M']

    print(f"预设配置: M={M_bsanfis}, 状态变量={len(state_features)}个, 解释变量={len(expl_features)}个")
    print(f"状态变量 (S): {state_features}")
    print(f"解释变量 (X): {expl_features}")

    # 2. 准备数据 (使用传入的、固定的训练/测试集索引)
    state_idx = [all_feature_names.index(f) for f in state_features]
    expl_idx = [all_feature_names.index(f) for f in expl_features]

    X_train_s = X_scaled_tensor[train_indices][:, state_idx].to(device)
    X_train_x = X_scaled_tensor[train_indices][:, expl_idx].to(device)
    y_train_t = y_scaled_tensor[train_indices].to(device)

    X_test_s = X_scaled_tensor[test_indices][:, state_idx].to(device)
    X_test_x = X_scaled_tensor[test_indices][:, expl_idx].to(device)
    y_test_t = y_scaled_tensor[test_indices]

    # 3. 训练模型
    bsanfis_model = SANFIS(membfuncs=create_membfuncs_config(len(state_features), M_bsanfis),
                           n_input=len(state_features)).to(device)
    optimizer_bsanfis = torch.optim.RMSprop(bsanfis_model.parameters(), lr=0.001)
    loss_fn_bsanfis = nn.MSELoss()

    for epoch in range(2000):
        bsanfis_model.train()
        optimizer_bsanfis.zero_grad()
        pred = bsanfis_model(X_train_s, X_train_x)
        loss = loss_fn_bsanfis(pred, y_train_t)
        loss.backward()
        optimizer_bsanfis.step()

    # 4. 在测试集上评估
    bsanfis_model.eval()
    with torch.no_grad():
        y_pred_scaled = bsanfis_model(X_test_s, X_test_x)

    y_pred_bsanfis = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy())
    y_test_orig = scaler_y.inverse_transform(y_test_t.cpu().numpy())

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_bsanfis)),
        'r2': r2_score(y_test_orig, y_pred_bsanfis),
        'mae': mean_absolute_error(y_test_orig, y_pred_bsanfis)
    }
    print(f"BS-ANFIS 性能: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
    return metrics


# ================================
# 对比模型定义
# ================================
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


# ================================
# 结果可视化函数
# ================================
def plot_comparison_results(results, save_dir):
    df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
    fig.suptitle('模型性能对比', fontsize=20, fontweight='bold')

    metrics = ['rmse', 'r2', 'mae']
    titles = ['均方根误差 (RMSE - 越低越好)', 'R²分数 (R-squared - 越高越好)', '平均绝对误差 (MAE - 越低越好)']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sorted_df = df.sort_values(metric, ascending=(metric != 'r2'))
        sns.barplot(x='Model', y=metric, data=sorted_df, ax=axes[i],
                    palette=sns.color_palette("viridis", n_colors=len(df)))
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel('模型', fontsize=12)
        axes[i].set_ylabel('值', fontsize=12)
        axes[i].tick_params(axis='x', rotation=15)

        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
        if metric == 'r2':
            axes[i].set_ylim(bottom=max(0, df['r2'].min() - 0.1), top=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, f'model_comparison_{timestamp}.png')
    plt.savefig(save_path)
    print(f"\n对比图已保存到: {save_path}")
    plt.show()


# ================================
# 主对比实验流程
# ================================
def run_comparison_experiment():
    print(f"\n{'=' * 80}")
    print("开始进行多模型对比实验 (BS-ANFIS使用预设参数)")
    print(f"{'=' * 80}")

    # 定义BS-ANFIS的预设最优参数
    best_params_for_bsanfis = {
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

    # 1. 统一数据划分
    all_indices = np.arange(X_scaled.shape[0])
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

    X_train_t = X_scaled_tensor[train_indices].to(device)
    y_train_t = y_scaled_tensor[train_indices].to(device)
    X_test_t = X_scaled_tensor[test_indices].to(device)
    y_test_orig = scaler_y.inverse_transform(y_scaled_tensor[test_indices].numpy())

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, _ = y_scaled[train_indices], y_scaled[test_indices]

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
    for _ in range(1000):
        ann_model.train()
        optimizer_ann.zero_grad()
        loss = loss_fn_ann(ann_model(X_train_t), y_train_t)
        loss.backward();
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
    M_anfis = 2
    print(f"标准ANFIS配置: M={M_anfis}, 输入特征={n_features}, 规则数={M_anfis ** n_features}")

    anfis_model = SANFIS(membfuncs=create_membfuncs_config(n_features, M_anfis), n_input=n_features).to(device)
    optimizer_anfis = torch.optim.RMSprop(anfis_model.parameters(), lr=0.001)
    loss_fn_anfis = nn.MSELoss()
    for _ in range(2000):
        anfis_model.train()
        optimizer_anfis.zero_grad()
        loss = loss_fn_anfis(anfis_model(X_train_t, X_train_t), y_train_t)
        loss.backward();
        optimizer_anfis.step()

    anfis_model.eval()
    with torch.no_grad():
        y_pred_anfis = scaler_y.inverse_transform(anfis_model(X_test_t, X_test_t).cpu().numpy())
    comparison_results['ANFIS'] = {
        'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_anfis)),
        'r2': r2_score(y_test_orig, y_pred_anfis),
        'mae': mean_absolute_error(y_test_orig, y_pred_anfis)
    }
    print(
        f"ANFIS 性能: RMSE={comparison_results['ANFIS']['rmse']:.4f}, R²={comparison_results['ANFIS']['r2']:.4f}, MAE={comparison_results['ANFIS']['mae']:.4f}")

    # --- 模型4: BS-ANFIS (使用预设参数) ---
    comparison_results['BS-ANFIS'] = run_preconfigured_bsanfis_and_get_results(best_params_for_bsanfis, train_indices,
                                                                               test_indices)

    # --- 最终结果汇总与可视化 ---
    print(f"\n{'=' * 80}")
    print("所有模型评估完成，生成最终对比图...")
    print(f"{'=' * 80}")
    plot_comparison_results(comparison_results, save_dir)


if __name__ == "__main__":
    run_comparison_experiment()
    print("\n所有实验和对比已成功完成！")
