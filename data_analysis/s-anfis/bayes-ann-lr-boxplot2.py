import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 创建保存目录
save_dir = 'sanfis_results/kfold_comparison'
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
# K折交叉验证结果可视化函数 (箱形图)
# ================================
def plot_kfold_boxplots(results_df, save_dir):
    palette = {'Train': '#00545E', 'Test': '#FDF5E6'}
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
    fig.suptitle('Model Performance Comparison (10-Fold Cross-Validation)', fontsize=20, fontweight='bold')

    # R² Score Plot
    ax_r2 = axes[0]
    sns.boxplot(data=results_df[results_df['Metric'] == 'R2'], x='Model', y='Value', hue='DataSet',
                palette=palette, ax=ax_r2, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': '6'},
                boxprops={'edgecolor': 'black'}, whiskerprops={'color': 'black'}, capprops={'color': 'black'},
                medianprops={'color': 'black'})
    ax_r2.set_title('(a) R² Score', fontsize=16, pad=10)
    ax_r2.set_ylabel('R²', fontsize=14)
    ax_r2.set_xlabel('Model', fontsize=14)
    ax_r2.set_ylim(bottom=max(0, results_df[results_df['Metric'] == 'R2']['Value'].min() - 0.1), top=1.01)
    ax_r2.grid(axis='y', linestyle='--', alpha=0.7)
    num_models = len(results_df['Model'].unique())
    for i in range(num_models - 1):
        ax_r2.axvline(i + 0.5, linestyle='--', color='grey')
    handles, labels = ax_r2.get_legend_handles_labels()
    ax_r2.legend(handles, ['Train', 'Test (Validation)'], title='Data Set', loc='lower left')

    # RMSE Plot
    ax_rmse = axes[1]
    sns.boxplot(data=results_df[results_df['Metric'] == 'RMSE'], x='Model', y='Value', hue='DataSet',
                palette=palette, ax=ax_rmse, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': '6'},
                boxprops={'edgecolor': 'black'}, whiskerprops={'color': 'black'}, capprops={'color': 'black'},
                medianprops={'color': 'black'})
    ax_rmse.set_title('(b) Root Mean Squared Error (RMSE)', fontsize=16, pad=10)
    ax_rmse.set_ylabel('RMSE (ppm)', fontsize=14)
    ax_rmse.set_xlabel('Model', fontsize=14)
    ax_rmse.set_ylim(bottom=0)
    ax_rmse.grid(axis='y', linestyle='--', alpha=0.7)
    for i in range(num_models - 1):
        ax_rmse.axvline(i + 0.5, linestyle='--', color='grey')
    ax_rmse.get_legend().remove()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path_png = os.path.join(save_dir, f'kfold_boxplot_comparison_{timestamp}.png')
    save_path_pdf = os.path.join(save_dir, f'kfold_boxplot_comparison_{timestamp}.pdf')
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    print(f"\nK折交叉验证对比箱形图已保存。")
    plt.show()


# ================================
# 样本外(OOF)预测散点图可视化函数
# ================================
def plot_oof_scatter_plots(oof_results, y_original, save_dir):
    n_models = len(oof_results)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 8 * n_rows), constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, (model_name, y_pred_oof) in enumerate(oof_results.items()):
        ax = axes[i]
        oof_r2 = r2_score(y_original, y_pred_oof)
        oof_rmse = np.sqrt(mean_squared_error(y_original, y_pred_oof))
        ax.scatter(y_original, y_pred_oof, alpha=0.7, color='darkorange', s=120, edgecolor='black',
                   label='Out-of-Fold Predictions')
        max_val = max(y_original.max(), y_pred_oof.max())
        margin = max_val * 0.05
        ax.set_xlim(0, max_val + margin);
        ax.set_ylim(0, max_val + margin)
        ax.plot([0, max_val + margin], [0, max_val + margin], 'k--', linewidth=2, label=r'Ideal Line $(y=x)$')
        textstr = (f'Overall Out-of-Fold Performance\n'
                   rf'$\mathrm{{R}}^2 = {oof_r2:.4f}$' + '\n' +
                   rf'$\mathrm{{RMSE}} = {oof_rmse:.4f}\,\mathrm{{ppm}}$')
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)
        ax.set_xlabel(r'Experimental $c_{\mathrm{H}_2\mathrm{S}}$ (ppm)', fontsize=20)
        ax.set_ylabel(r'Predicted $c_{\mathrm{H}_2\mathrm{S}}$ (ppm)', fontsize=20)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', direction='out', length=6, color='black', labelsize=16)
        letters = ['a', 'b', 'c', 'd']
        ax.set_title(f'{letters[i]}. {model_name}', fontsize=22, fontweight='bold', pad=10)
        ax.legend(loc='lower right', fontsize=18)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('black');
            spine.set_linewidth(1.0)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'oof_prediction_scatter_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'oof_prediction_scatter_{timestamp}.pdf'), bbox_inches='tight')
    print(f"OOF散点对比图已保存。")
    plt.show()


# ================================
# 主对比实验流程
# ================================
def run_comparison_experiment_kfold():
    print(f"\n{'=' * 80}")
    print("开始进行多模型对比实验 (10折交叉验证)")
    print(f"{'=' * 80}")

    N_SPLITS = 100
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    all_results = []

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

    oof_preds = {
        'Linear Regression (LR)': np.zeros_like(y_full),
        'Neural Network (ANN)': np.zeros_like(y_full),
        'Standard ANFIS': np.zeros_like(y_full),
        'Optimized BS-ANFIS': np.zeros_like(y_full)
    }

    for fold, (train_indices, test_indices) in enumerate(kf.split(X_scaled)):
        print(f"\n{'=' * 20} Fold {fold + 1}/{N_SPLITS} {'=' * 20}")
        X_train_t, y_train_t = X_scaled_tensor[train_indices].to(device), y_scaled_tensor[train_indices].to(device)
        X_test_t = X_scaled_tensor[test_indices].to(device)
        X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
        y_train_scaled, y_test_scaled = y_scaled[train_indices], y_scaled[test_indices]
        y_train_orig, y_test_orig = scaler_y.inverse_transform(y_train_scaled), scaler_y.inverse_transform(
            y_test_scaled)

        # --- 模型1: 线性回归 (LR) ---
        print("Training and evaluating: 1. Linear Regression (LR)")
        lr_model = LinearRegression().fit(X_train, y_train_scaled)
        y_pred_train_lr = scaler_y.inverse_transform(lr_model.predict(X_train))
        y_pred_test_lr = scaler_y.inverse_transform(lr_model.predict(X_test))
        oof_preds['Linear Regression (LR)'][test_indices] = y_pred_test_lr
        for dataset, y_true, y_pred in [('Train', y_train_orig, y_pred_train_lr),
                                        ('Test', y_test_orig, y_pred_test_lr)]:
            all_results.append({'Fold': fold, 'Model': 'LR', 'DataSet': dataset, 'Metric': 'RMSE',
                                'Value': np.sqrt(mean_squared_error(y_true, y_pred))})
            all_results.append(
                {'Fold': fold, 'Model': 'LR', 'DataSet': dataset, 'Metric': 'R2', 'Value': r2_score(y_true, y_pred)})

        # --- 模型2: 人工神经网络 (ANN) ---
        print("Training and evaluating: 2. Artificial Neural Network (ANN)")
        ann_model = ANN(input_size=X_train.shape[1]).to(device)
        optimizer_ann = torch.optim.Adam(ann_model.parameters(), lr=0.01)
        loss_fn_ann = nn.MSELoss()
        for _ in range(500):
            ann_model.train();
            optimizer_ann.zero_grad()
            loss = loss_fn_ann(ann_model(X_train_t), y_train_t)
            loss.backward();
            optimizer_ann.step()
        ann_model.eval()
        with torch.no_grad():
            y_pred_train_ann = scaler_y.inverse_transform(ann_model(X_train_t).cpu().numpy())
            y_pred_test_ann = scaler_y.inverse_transform(ann_model(X_test_t).cpu().numpy())
        oof_preds['Neural Network (ANN)'][test_indices] = y_pred_test_ann
        for dataset, y_true, y_pred in [('Train', y_train_orig, y_pred_train_ann),
                                        ('Test', y_test_orig, y_pred_test_ann)]:
            all_results.append({'Fold': fold, 'Model': 'ANN', 'DataSet': dataset, 'Metric': 'RMSE',
                                'Value': np.sqrt(mean_squared_error(y_true, y_pred))})
            all_results.append(
                {'Fold': fold, 'Model': 'ANN', 'DataSet': dataset, 'Metric': 'R2', 'Value': r2_score(y_true, y_pred)})

        # --- 模型3: 标准 ANFIS ---
        print("Training and evaluating: 3. Standard ANFIS")
        anfis_model = SANFIS(membfuncs=create_membfuncs_config(X_train.shape[1], 3), n_input=X_train.shape[1]).to(
            device)
        optimizer_anfis = torch.optim.RMSprop(anfis_model.parameters(), lr=0.001)
        loss_fn_anfis = nn.MSELoss()
        for _ in range(2500):
            anfis_model.train();
            optimizer_anfis.zero_grad()
            loss = loss_fn_anfis(anfis_model(X_train_t, X_train_t), y_train_t)
            loss.backward();
            optimizer_anfis.step()
        anfis_model.eval()
        with torch.no_grad():
            y_pred_train_anfis = scaler_y.inverse_transform(anfis_model(X_train_t, X_train_t).cpu().numpy())
            y_pred_test_anfis = scaler_y.inverse_transform(anfis_model(X_test_t, X_test_t).cpu().numpy())
        oof_preds['Standard ANFIS'][test_indices] = y_pred_test_anfis
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
        for _ in range(2500):
            bsanfis_model.train();
            optimizer_bsanfis.zero_grad()
            loss = loss_fn_bsanfis(bsanfis_model(X_train_s, X_train_x), y_train_t)
            loss.backward();
            optimizer_bsanfis.step()
        bsanfis_model.eval()
        with torch.no_grad():
            y_pred_train_bsanfis = scaler_y.inverse_transform(bsanfis_model(X_train_s, X_train_x).cpu().numpy())
            y_pred_test_bsanfis = scaler_y.inverse_transform(bsanfis_model(X_test_s, X_test_x).cpu().numpy())
        oof_preds['Optimized BS-ANFIS'][test_indices] = y_pred_test_bsanfis
        for dataset, y_true, y_pred in [('Train', y_train_orig, y_pred_train_bsanfis),
                                        ('Test', y_test_orig, y_pred_test_bsanfis)]:
            all_results.append({'Fold': fold, 'Model': 'BS-ANFIS', 'DataSet': dataset, 'Metric': 'RMSE',
                                'Value': np.sqrt(mean_squared_error(y_true, y_pred))})
            all_results.append({'Fold': fold, 'Model': 'BS-ANFIS', 'DataSet': dataset, 'Metric': 'R2',
                                'Value': r2_score(y_true, y_pred)})

    # --- 最终结果汇总、保存与可视化 ---
    print(f"\n{'=' * 80}\n所有折训练完成，正在汇总结果并保存...\n{'=' * 80}")

    results_df = pd.DataFrame(all_results)
    summary_df = results_df[results_df['DataSet'] == 'Test'].groupby(['Model', 'Metric'])['Value'].agg(
        ['mean', 'std']).reset_index()
    print("10折交叉验证测试集平均性能总结:")
    print(summary_df.to_string())

    oof_df = pd.DataFrame(y_full, columns=['y_true'])
    for model_name, preds in oof_preds.items():
        oof_df[f'y_pred_{model_name.replace(" ", "_")}'] = preds

        # 4. 将所有结果保存到一个Excel文件中
    excel_save_path = os.path.join(save_dir, f'kfold_results_{timestamp}.xlsx')
    print(f"\n正在将所有结果保存到: {excel_save_path}")
    with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
        oof_df.to_excel(writer, sheet_name='OOF_Predictions', index=False)
        results_df.to_excel(writer, sheet_name='All_Fold_Metrics', index=False)
    print("Excel文件保存成功！")

    print("\n正在生成可视化图表...")
    model_name_map = {'LR': 'LR', 'ANN': 'ANN', 'ANFIS': 'ANFIS', 'BS-ANFIS': 'BS-ANFIS'}
    results_df['Model'] = results_df['Model'].map(model_name_map)

    plot_kfold_boxplots(results_df, save_dir)
    plt.close('all')

    plot_oof_scatter_plots(oof_preds, y_full, save_dir)
    plt.close('all')


if __name__ == "__main__":
    run_comparison_experiment_kfold()
    print("\n所有实验、保存和对比已成功完成！")
