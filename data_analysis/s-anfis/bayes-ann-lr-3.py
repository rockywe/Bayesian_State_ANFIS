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
save_dir = 'sanfis_results/final_detailed_comparison'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ================================
# 数据加载与预处理 (代码不变)
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
# 辅助函数 (代码不变)
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
# 对比模型定义 (代码不变)
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
def plot_summary_comparison(results, save_dir):
    df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
    fig.suptitle('模型性能汇总对比', fontsize=20, fontweight='bold')
    metrics = ['rmse', 'r2', 'mae']
    titles = ['均方根误差 (RMSE - 越低越好)', 'R²分数 (R-squared - 越高越好)', '平均绝对误差 (MAE - 越低越好)']
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sorted_df = df.sort_values(metric, ascending=(metric != 'r2'))
        sns.barplot(x='Model', y=metric, data=sorted_df, ax=axes[i],
                    palette=sns.color_palette("viridis", n_colors=len(df)))
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel('模型', fontsize=12);
        axes[i].set_ylabel('值', fontsize=12)
        axes[i].tick_params(axis='x', rotation=15)
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
        if metric == 'r2': axes[i].set_ylim(bottom=max(0, df['r2'].min() - 0.1), top=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, f'summary_comparison_{timestamp}.png')
    plt.savefig(save_path)
    print(f"\n汇总对比图已保存到: {save_path}")
    plt.show()


### 新增部分：详细散点图可视化函数 ###
def plot_detailed_scatter_plots(detailed_results, save_dir):
    """
    为每个模型生成详细的散点图，包含训练集和测试集数据。
    """
    n_models = len(detailed_results)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 8 * n_rows), constrained_layout=True)
    axes = np.array(axes).flatten()  # 确保axes总是一维数组，方便索引

    for i, result in enumerate(detailed_results):
        ax = axes[i]

        # 提取数据
        y_train_orig = result['y_train_original']
        y_pred_train = result['y_pred_train']
        y_test_orig = result['y_test_original']
        y_pred_test = result['y_pred_test']

        # 计算测试集指标
        r2 = r2_score(y_test_orig, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test))

        # 绘制散点图
        ax.scatter(y_train_orig, y_pred_train, alpha=0.6, color='royalblue', s=50, label='Train Set')
        ax.scatter(y_test_orig, y_pred_test, alpha=0.9, color='darkorange', s=80, edgecolor='black', label='Test Set')

        # 设置轴范围
        max_val = max(y_train_orig.max(), y_pred_train.max(), y_test_orig.max(), y_pred_test.max())
        margin = max_val * 0.05
        ax.set_xlim(0, max_val + margin)
        ax.set_ylim(0, max_val + margin)

        # 绘制理想线
        ax.plot([0, max_val + margin], [0, max_val + margin], 'k--', linewidth=2, label='Ideal Line (y=x)')

        # 添加性能指标文本框 (基于测试集)
        textstr = (rf'$\mathrm{{R}}^2_{{test}} = {r2:.3f}$' + '\n' +
                   rf'$\mathrm{{RMSE}}_{{test}} = {rmse:.3f}$')
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        # 设置标签、标题和图例
        ax.set_xlabel('Experimental Value', fontsize=16)
        ax.set_ylabel('Predicted Value', fontsize=16)
        ax.set_title(result['model_name'], fontsize=18, fontweight='bold')
        ax.legend(loc='lower right', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Model Prediction Performance (Train vs. Test)', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path_png = os.path.join(save_dir, f'detailed_scatter_plots_{timestamp}.png')
    plt.savefig(save_path_png, dpi=300)
    print(f"详细散点对比图已保存到: {save_path_png}")
    plt.show()


# ================================
# 主对比实验流程
# ================================
def run_comparison_experiment():
    print(f"\n{'=' * 80}")
    print("开始进行多模型对比实验 (BS-ANFIS使用预设参数)")
    print(f"{'=' * 80}")

    best_params_for_bsanfis = {
        "role_GasInletFlow": "both", "role_GasInletTemperature": "explanatory",
        "role_GasInletPressure": "state", "role_DesulfurizationLiquidFlow": "state",
        "role_DesulfurizationLiquidTemperature": "state", "role_DesulfurizationLiquidPressure": "explanatory",
        "role_RotationSpeed": "both", "role_HtwoSInletConcentration": "explanatory", "M": 3
    }

    all_indices = np.arange(X_scaled.shape[0])
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

    # 准备数据
    X_train_t = X_scaled_tensor[train_indices].to(device)
    y_train_t = y_scaled_tensor[train_indices].to(device)
    X_test_t = X_scaled_tensor[test_indices].to(device)

    y_train_orig = scaler_y.inverse_transform(y_scaled[train_indices])
    y_test_orig = scaler_y.inverse_transform(y_scaled[test_indices])

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train_scaled, _ = y_scaled[train_indices], y_scaled[test_indices]

    summary_results = {}
    detailed_results_list = []

    # --- 模型1: 线性回归 (LR) ---
    print("\n--- 训练与评估: 1. 线性回归 (LR) ---")
    lr_model = LinearRegression().fit(X_train, y_train_scaled)
    y_pred_train_lr = scaler_y.inverse_transform(lr_model.predict(X_train))
    y_pred_test_lr = scaler_y.inverse_transform(lr_model.predict(X_test))
    summary_results['LR'] = {'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_test_lr)),
                             'r2': r2_score(y_test_orig, y_pred_test_lr),
                             'mae': mean_absolute_error(y_test_orig, y_pred_test_lr)}
    detailed_results_list.append(
        {'model_name': 'Linear Regression (LR)', 'y_train_original': y_train_orig, 'y_pred_train': y_pred_train_lr,
         'y_test_original': y_test_orig, 'y_pred_test': y_pred_test_lr})
    print(f"LR 性能(测试集): RMSE={summary_results['LR']['rmse']:.4f}, R²={summary_results['LR']['r2']:.4f}")

    # --- 模型2: 人工神经网络 (ANN) ---
    print("\n--- 训练与评估: 2. 人工神经网络 (ANN) ---")
    ann_model = ANN(input_size=X_train.shape[1]).to(device)
    optimizer_ann = torch.optim.RMSprop(ann_model.parameters(), lr=0.01)
    loss_fn_ann = nn.MSELoss()
    for _ in range(800):
        ann_model.train();
        optimizer_ann.zero_grad()
        loss = loss_fn_ann(ann_model(X_train_t), y_train_t)
        loss.backward();
        optimizer_ann.step()

    ann_model.eval()
    with torch.no_grad():
        y_pred_train_ann = scaler_y.inverse_transform(ann_model(X_train_t).cpu().numpy())
        y_pred_test_ann = scaler_y.inverse_transform(ann_model(X_test_t).cpu().numpy())
    summary_results['ANN'] = {'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_test_ann)),
                              'r2': r2_score(y_test_orig, y_pred_test_ann),
                              'mae': mean_absolute_error(y_test_orig, y_pred_test_ann)}
    detailed_results_list.append(
        {'model_name': 'Neural Network (ANN)', 'y_train_original': y_train_orig, 'y_pred_train': y_pred_train_ann,
         'y_test_original': y_test_orig, 'y_pred_test': y_pred_test_ann})
    print(f"ANN 性能(测试集): RMSE={summary_results['ANN']['rmse']:.4f}, R²={summary_results['ANN']['r2']:.4f}")

    # --- 模型3: 标准 ANFIS ---
    print("\n--- 训练与评估: 3. 标准 ANFIS ---")
    n_features, M_anfis = X_train.shape[1], 2
    anfis_model = SANFIS(membfuncs=create_membfuncs_config(n_features, M_anfis), n_input=n_features).to(device)
    optimizer_anfis = torch.optim.RMSprop(anfis_model.parameters(), lr=0.001)
    loss_fn_anfis = nn.MSELoss()
    for _ in range(2000):
        anfis_model.train();
        optimizer_anfis.zero_grad()
        loss = loss_fn_anfis(anfis_model(X_train_t, X_train_t), y_train_t)
        loss.backward();
        optimizer_anfis.step()

    anfis_model.eval()
    with torch.no_grad():
        y_pred_train_anfis = scaler_y.inverse_transform(anfis_model(X_train_t, X_train_t).cpu().numpy())
        y_pred_test_anfis = scaler_y.inverse_transform(anfis_model(X_test_t, X_test_t).cpu().numpy())
    summary_results['ANFIS'] = {'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_test_anfis)),
                                'r2': r2_score(y_test_orig, y_pred_test_anfis),
                                'mae': mean_absolute_error(y_test_orig, y_pred_test_anfis)}
    detailed_results_list.append(
        {'model_name': 'Standard ANFIS', 'y_train_original': y_train_orig, 'y_pred_train': y_pred_train_anfis,
         'y_test_original': y_test_orig, 'y_pred_test': y_pred_test_anfis})
    print(f"ANFIS 性能(测试集): RMSE={summary_results['ANFIS']['rmse']:.4f}, R²={summary_results['ANFIS']['r2']:.4f}")

    # --- 模型4: BS-ANFIS (使用预设参数) ---
    print("\n--- 训练与评估: 4. BS-ANFIS (使用预设最优配置) ---")
    config_map = {'GasInletFlow': 'Gas_Inlet_Flow', 'GasInletTemperature': 'Gas_Inlet_Temperature',
                  'GasInletPressure': 'Gas_Inlet_Pressure', 'DesulfurizationLiquidFlow': 'Desulfurization_Liquid_Flow',
                  'DesulfurizationLiquidTemperature': 'Desulfurization_Liquid_Temperature',
                  'DesulfurizationLiquidPressure': 'Desulfurization_Liquid_Pressure', 'RotationSpeed': 'Rotation_Speed',
                  'HtwoSInletConcentration': 'H2S_Inlet_Concentration'}
    s_features, x_features = [], []
    for k, r in best_params_for_bsanfis.items():
        if k.startswith("role_"):
            col = config_map.get(k.replace("role_", ""))
            if col:
                if r == 'state':
                    s_features.append(col)
                elif r == 'explanatory':
                    x_features.append(col)
                elif r == 'both':
                    s_features.append(col); x_features.append(col)
    s_features, x_features = sorted(list(set(s_features))), sorted(list(set(x_features)))
    s_idx, x_idx = [all_feature_names.index(f) for f in s_features], [all_feature_names.index(f) for f in x_features]

    X_train_s, X_train_x = X_train_t[:, s_idx], X_train_t[:, x_idx]
    X_test_s, X_test_x = X_test_t[:, s_idx], X_test_t[:, x_idx]

    bsanfis_model = SANFIS(membfuncs=create_membfuncs_config(len(s_features), best_params_for_bsanfis['M']),
                           n_input=len(s_features)).to(device)
    optimizer_bsanfis = torch.optim.RMSprop(bsanfis_model.parameters(), lr=0.001)
    loss_fn_bsanfis = nn.MSELoss()
    for _ in range(2000):
        bsanfis_model.train();
        optimizer_bsanfis.zero_grad()
        loss = loss_fn_bsanfis(bsanfis_model(X_train_s, X_train_x), y_train_t)
        loss.backward();
        optimizer_bsanfis.step()

    bsanfis_model.eval()
    with torch.no_grad():
        y_pred_train_bsanfis = scaler_y.inverse_transform(bsanfis_model(X_train_s, X_train_x).cpu().numpy())
        y_pred_test_bsanfis = scaler_y.inverse_transform(bsanfis_model(X_test_s, X_test_x).cpu().numpy())
    summary_results['BS-ANFIS'] = {'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_test_bsanfis)),
                                   'r2': r2_score(y_test_orig, y_pred_test_bsanfis),
                                   'mae': mean_absolute_error(y_test_orig, y_pred_test_bsanfis)}
    detailed_results_list.append(
        {'model_name': 'Optimized BS-ANFIS', 'y_train_original': y_train_orig, 'y_pred_train': y_pred_train_bsanfis,
         'y_test_original': y_test_orig, 'y_pred_test': y_pred_test_bsanfis})
    print(
        f"BS-ANFIS 性能(测试集): RMSE={summary_results['BS-ANFIS']['rmse']:.4f}, R²={summary_results['BS-ANFIS']['r2']:.4f}")

    # --- 最终结果汇总与可视化 ---
    print(f"\n{'=' * 80}\n生成最终对比图...\n{'=' * 80}")
    plot_summary_comparison(summary_results, save_dir)
    plot_detailed_scatter_plots(detailed_results_list, save_dir)


if __name__ == "__main__":
    run_comparison_experiment()
    print("\n所有实验和对比已成功完成！")

