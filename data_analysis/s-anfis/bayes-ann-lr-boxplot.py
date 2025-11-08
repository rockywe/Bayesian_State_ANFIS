import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold  # 引入KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import os
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
is_available = torch.cuda.is_available()
print(f"CUDA available: {is_available}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 创建保存目录
save_dir = 'sanfis_results/kfold_comparison'
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
            nn.Linear(input_size, hidden_size1), nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size2), nn.Tanh(),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        return self.network(x)


# ================================
# 新增：K折交叉验证结果可视化函数
# ================================
def plot_kfold_boxplots(results_df, save_dir):
    """
    根据K折交叉验证的结果绘制箱形图，风格类似于示例图片。
    """
    # 定义颜色以匹配示例图中的 Train 和 Validation/Test
    # 我们将交叉验证中的测试集视为 'Validation'
    palette = {
        'Train': '#00545E',  # 深青色
        'Test': '#FF8C00',  # 橙色 (对应示例图中的 Test)
        'Validation': '#FDF5E6'  # 米白色 (对应示例图中的 Validation)
    }

    # 为了简化，我们只生成Train和Test(Validation)两个分布
    # 如果需要三个，需要在K-Fold循环内再做一次划分
    plot_palette = {
        'Train': palette['Train'],
        'Test': palette['Validation']  # 使用Validation的颜色来表示CV中的Test集
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
    fig.suptitle('Model Performance Comparison (10-Fold Cross-Validation)', fontsize=20, fontweight='bold')

    # --- (a) R² Score Plot ---
    ax_r2 = axes[0]
    sns.boxplot(data=results_df[results_df['Metric'] == 'R2'], x='Model', y='Value', hue='DataSet',
                palette=plot_palette, ax=ax_r2, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': '6'},
                boxprops={'edgecolor': 'black'}, whiskerprops={'color': 'black'}, capprops={'color': 'black'},
                medianprops={'color': 'black'})

    ax_r2.set_title('(a) R² Score', fontsize=16, pad=10)
    ax_r2.set_ylabel('R²', fontsize=14)
    ax_r2.set_xlabel('Model', fontsize=14)
    ax_r2.set_ylim(bottom=max(0, results_df[results_df['Metric'] == 'R2']['Value'].min() - 0.1), top=1.01)
    ax_r2.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加垂直分隔线
    num_models = len(results_df['Model'].unique())
    for i in range(num_models - 1):
        ax_r2.axvline(i + 0.5, linestyle='--', color='grey')

    # 自定义图例
    handles, labels = ax_r2.get_legend_handles_labels()
    ax_r2.legend(handles, ['Train', 'Test (Validation)'], title='Data Set', loc='lower left')

    # --- (b) RMSE Plot ---
    ax_rmse = axes[1]
    sns.boxplot(data=results_df[results_df['Metric'] == 'RMSE'], x='Model', y='Value', hue='DataSet',
                palette=plot_palette, ax=ax_rmse, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': '6'},
                boxprops={'edgecolor': 'black'}, whiskerprops={'color': 'black'}, capprops={'color': 'black'},
                medianprops={'color': 'black'})

    ax_rmse.set_title('(b) Root Mean Squared Error (RMSE)', fontsize=16, pad=10)
    ax_rmse.set_ylabel('RMSE (ppm)', fontsize=14)
    ax_rmse.set_xlabel('Model', fontsize=14)
    ax_rmse.set_ylim(bottom=0)
    ax_rmse.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加垂直分隔线
    for i in range(num_models - 1):
        ax_rmse.axvline(i + 0.5, linestyle='--', color='grey')

    ax_rmse.get_legend().remove()  # 移除第二个图的图例以保持整洁

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path_png = os.path.join(save_dir, f'kfold_boxplot_comparison_{timestamp}.png')
    save_path_pdf = os.path.join(save_dir, f'kfold_boxplot_comparison_{timestamp}.pdf')
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    print(f"\nK折交叉验证对比箱形图已保存到: {save_path_png} 和 {save_path_pdf}")
    plt.show()


# ================================
# 主对比实验流程 (已重构为K折交叉验证)
# ================================
def run_comparison_experiment_kfold():
    print(f"\n{'=' * 80}")
    print("开始进行多模型对比实验 (k折交叉验证)")
    print(f"{'=' * 80}")

    # K折交叉验证设置
    N_SPLITS = 4
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 用于存储所有折叠、所有模型结果的列表
    all_results = []

    # BS-ANFIS 的固定参数
    best_params_for_bsanfis = {
        "role_GasInletFlow": "both", "role_GasInletTemperature": "explanatory",
        "role_GasInletPressure": "state", "role_DesulfurizationLiquidFlow": "state",
        "role_DesulfurizationLiquidTemperature": "state", "role_DesulfurizationLiquidPressure": "explanatory",
        "role_RotationSpeed": "both", "role_HtwoSInletConcentration": "explanatory", "M": 3
    }
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

    # 开始K折交叉验证循环
    for fold, (train_indices, test_indices) in enumerate(kf.split(X_scaled)):
        print(f"\n{'=' * 20} Fold {fold + 1}/{N_SPLITS} {'=' * 20}")

        # 准备当前折的数据
        X_train_t = X_scaled_tensor[train_indices].to(device)
        y_train_t = y_scaled_tensor[train_indices].to(device)
        X_test_t = X_scaled_tensor[test_indices].to(device)
        y_test_t = y_scaled_tensor[test_indices].to(device)  # 需要用于计算测试集损失

        # 用于评估的Numpy数据
        X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
        y_train_scaled, y_test_scaled = y_scaled[train_indices], y_scaled[test_indices]
        y_train_orig = scaler_y.inverse_transform(y_train_scaled)
        y_test_orig = scaler_y.inverse_transform(y_test_scaled)

        # --- 模型1: 线性回归 (LR) ---
        print("Training and evaluating: 1. Linear Regression (LR)")
        lr_model = LinearRegression().fit(X_train, y_train_scaled)
        y_pred_train_lr = scaler_y.inverse_transform(lr_model.predict(X_train))
        y_pred_test_lr = scaler_y.inverse_transform(lr_model.predict(X_test))

        for dataset, y_true, y_pred in [('Train', y_train_orig, y_pred_train_lr),
                                        ('Test', y_test_orig, y_pred_test_lr)]:
            all_results.append({'Fold': fold, 'Model': 'LR', 'DataSet': dataset, 'Metric': 'RMSE',
                                'Value': np.sqrt(mean_squared_error(y_true, y_pred))})
            all_results.append(
                {'Fold': fold, 'Model': 'LR', 'DataSet': dataset, 'Metric': 'R2', 'Value': r2_score(y_true, y_pred)})

        # --- 模型2: 人工神经网络 (ANN) ---
        print("Training and evaluating: 2. Artificial Neural Network (ANN)")
        ann_model = ANN(input_size=X_train.shape[1]).to(device)
        optimizer_ann = torch.optim.Adam(ann_model.parameters(), lr=0.001)
        loss_fn_ann = nn.MSELoss()
        for _ in range(1500):
            ann_model.train();
            optimizer_ann.zero_grad()
            loss = loss_fn_ann(ann_model(X_train_t), y_train_t)
            loss.backward();
            optimizer_ann.step()

        ann_model.eval()
        with torch.no_grad():
            y_pred_train_ann = scaler_y.inverse_transform(ann_model(X_train_t).cpu().numpy())
            y_pred_test_ann = scaler_y.inverse_transform(ann_model(X_test_t).cpu().numpy())
        for dataset, y_true, y_pred in [('Train', y_train_orig, y_pred_train_ann),
                                        ('Test', y_test_orig, y_pred_test_ann)]:
            all_results.append({'Fold': fold, 'Model': 'ANN', 'DataSet': dataset, 'Metric': 'RMSE',
                                'Value': np.sqrt(mean_squared_error(y_true, y_pred))})
            all_results.append(
                {'Fold': fold, 'Model': 'ANN', 'DataSet': dataset, 'Metric': 'R2', 'Value': r2_score(y_true, y_pred)})

        # --- 模型3: 标准 ANFIS ---
        print("Training and evaluating: 3. Standard ANFIS")
        n_features, M_anfis = X_train.shape[1], 3
        anfis_model = SANFIS(membfuncs=create_membfuncs_config(n_features, M_anfis), n_input=n_features).to(device)
        optimizer_anfis = torch.optim.RMSprop(anfis_model.parameters(), lr=0.001)
        loss_fn_anfis = nn.MSELoss()
        for _ in range(3000):
            anfis_model.train();
            optimizer_anfis.zero_grad()
            loss = loss_fn_anfis(anfis_model(X_train_t, X_train_t), y_train_t)
            loss.backward();
            optimizer_anfis.step()

        anfis_model.eval()
        with torch.no_grad():
            y_pred_train_anfis = scaler_y.inverse_transform(anfis_model(X_train_t, X_train_t).cpu().numpy())
            y_pred_test_anfis = scaler_y.inverse_transform(anfis_model(X_test_t, X_test_t).cpu().numpy())
        for dataset, y_true, y_pred in [('Train', y_train_orig, y_pred_train_anfis),
                                        ('Test', y_test_orig, y_pred_test_anfis)]:
            all_results.append({'Fold': fold, 'Model': 'ANFIS', 'DataSet': dataset, 'Metric': 'RMSE',
                                'Value': np.sqrt(mean_squared_error(y_true, y_pred))})
            all_results.append(
                {'Fold': fold, 'Model': 'ANFIS', 'DataSet': dataset, 'Metric': 'R2', 'Value': r2_score(y_true, y_pred)})

        # --- 模型4: BS-ANFIS (使用预设参数) ---
        print("Training and evaluating: 4. BS-ANFIS (Optimized)")
        X_train_s, X_train_x = X_train_t[:, s_idx], X_train_t[:, x_idx]
        X_test_s, X_test_x = X_test_t[:, s_idx], X_test_t[:, x_idx]
        bsanfis_model = SANFIS(membfuncs=create_membfuncs_config(len(s_features), best_params_for_bsanfis['M']),
                               n_input=len(s_features)).to(device)
        optimizer_bsanfis = torch.optim.RMSprop(bsanfis_model.parameters(), lr=0.001)
        loss_fn_bsanfis = nn.MSELoss()
        for _ in range(3000):
            bsanfis_model.train();
            optimizer_bsanfis.zero_grad()
            loss = loss_fn_bsanfis(bsanfis_model(X_train_s, X_train_x), y_train_t)
            loss.backward();
            optimizer_bsanfis.step()

        bsanfis_model.eval()
        with torch.no_grad():
            y_pred_train_bsanfis = scaler_y.inverse_transform(bsanfis_model(X_train_s, X_train_x).cpu().numpy())
            y_pred_test_bsanfis = scaler_y.inverse_transform(bsanfis_model(X_test_s, X_test_x).cpu().numpy())
        for dataset, y_true, y_pred in [('Train', y_train_orig, y_pred_train_bsanfis),
                                        ('Test', y_test_orig, y_pred_test_bsanfis)]:
            all_results.append({'Fold': fold, 'Model': 'BS-ANFIS', 'DataSet': dataset, 'Metric': 'RMSE',
                                'Value': np.sqrt(mean_squared_error(y_true, y_pred))})
            all_results.append({'Fold': fold, 'Model': 'BS-ANFIS', 'DataSet': dataset, 'Metric': 'R2',
                                'Value': r2_score(y_true, y_pred)})

    # --- 最终结果汇总与可视化 ---
    print(f"\n{'=' * 80}\n所有折训练完成，正在生成最终对比图...\n{'=' * 80}")

    # 将结果列表转换为DataFrame
    results_df = pd.DataFrame(all_results)

    # 打印测试集上的平均性能
    summary = results_df[results_df['DataSet'] == 'Test'].groupby(['Model', 'Metric'])['Value'].agg(
        ['mean', 'std']).reset_index()
    print("10折交叉验证测试集平均性能总结:")
    print(summary.to_string())

    # 更改模型名称以适应图例
    model_name_map = {
        'LR': 'LR',
        'ANN': 'ANN',
        'ANFIS': 'ANFIS',
        'BS-ANFIS': 'BS-ANFIS'
    }
    results_df['Model'] = results_df['Model'].map(model_name_map)

    # 绘制箱形图
    plot_kfold_boxplots(results_df, save_dir)


if __name__ == "__main__":
    # 运行K折交叉验证实验
    run_comparison_experiment_kfold()
    print("\n所有实验和对比已成功完成！")
