# 文件名: run_stat_tests.py
# 描述: 此脚本仅执行模型训练和统计检验，已移除所有绘图功能。

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sanfis import SANFIS
from scipy import stats  # 导入统计检验库


# ================================
# 全局设备配置
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

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
# 主实验与检验流程
# ================================
def run_experiment_and_tests():
    print(f"\n{'=' * 80}")
    print("开始进行模型训练、评估与统计检验")
    print(f"{'=' * 80}")

    best_params_for_bsanfis = {
        "role_GasInletFlow": "both", "role_GasInletTemperature": "explanatory",
        "role_GasInletPressure": "state", "role_DesulfurizationLiquidFlow": "state",
        "role_DesulfurizationLiquidTemperature": "state", "role_DesulfurizationLiquidPressure": "explanatory",
        "role_RotationSpeed": "both", "role_HtwoSInletConcentration": "explanatory", "M": 3
    }

    all_indices = np.arange(X_scaled.shape[0])
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

    X_train_t, y_train_t = X_scaled_tensor[train_indices].to(device), y_scaled_tensor[train_indices].to(device)
    X_test_t = X_scaled_tensor[test_indices].to(device)
    y_train_orig, y_test_orig = scaler_y.inverse_transform(y_scaled[train_indices]), scaler_y.inverse_transform(
        y_scaled[test_indices])
    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train_scaled, _ = y_scaled[train_indices], y_scaled[test_indices]

    # detailed_results_list 用于存储每个模型的预测结果，以供后续统计检验使用
    detailed_results_list = []

    # --- 模型1: 线性回归 (LR) ---
    print("\n--- 训练与评估: 1. 线性回归 (LR) ---")
    lr_model = LinearRegression().fit(X_train, y_train_scaled)
    y_pred_test_lr = scaler_y.inverse_transform(lr_model.predict(X_test))
    rmse_lr = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_lr))
    r2_lr = r2_score(y_test_orig, y_pred_test_lr)
    detailed_results_list.append({'model_name': 'LR', 'y_test_original': y_test_orig, 'y_pred_test': y_pred_test_lr})
    print(f"LR 性能(测试集): RMSE={rmse_lr:.4f}, R²={r2_lr:.4f}")

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
        y_pred_test_ann = scaler_y.inverse_transform(ann_model(X_test_t).cpu().numpy())
    rmse_ann = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_ann))
    r2_ann = r2_score(y_test_orig, y_pred_test_ann)
    detailed_results_list.append({'model_name': 'ANN', 'y_test_original': y_test_orig, 'y_pred_test': y_pred_test_ann})
    print(f"ANN 性能(测试集): RMSE={rmse_ann:.4f}, R²={r2_ann:.4f}")

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
        y_pred_test_anfis = scaler_y.inverse_transform(anfis_model(X_test_t, X_test_t).cpu().numpy())
    rmse_anfis = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_anfis))
    r2_anfis = r2_score(y_test_orig, y_pred_test_anfis)
    detailed_results_list.append(
        {'model_name': 'ANFIS', 'y_test_original': y_test_orig, 'y_pred_test': y_pred_test_anfis})
    print(f"ANFIS 性能(测试集): RMSE={rmse_anfis:.4f}, R²={r2_anfis:.4f}")

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
        y_pred_test_bsanfis = scaler_y.inverse_transform(bsanfis_model(X_test_s, X_test_x).cpu().numpy())
    rmse_bsanfis = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_bsanfis))
    r2_bsanfis = r2_score(y_test_orig, y_pred_test_bsanfis)
    detailed_results_list.append(
        {'model_name': 'BS-ANFIS', 'y_test_original': y_test_orig, 'y_pred_test': y_pred_test_bsanfis})
    print(f"BS-ANFIS 性能(测试集): RMSE={rmse_bsanfis:.4f}, R²={r2_bsanfis:.4f}")

    # =========================================================================
    # 统计学检验
    # =========================================================================
    print(f"\n\n{'=' * 80}")
    print("最终统计学检验结果")
    print(f"{'=' * 80}")

    # --- 1. Kruskal-Wallis H检验 ---
    print("\n--- 表4: Kruskal-Wallis H检验 (模型可靠性) ---")
    print("H0: 预测值与实验值之间没有显著差异。 P > 0.05 则接受H0。\n")

    kruskal_results = []
    for result in detailed_results_list:
        model_name = result['model_name']
        y_true = result['y_test_original'].flatten()
        y_pred = result['y_pred_test'].flatten()

        stat, p_value = stats.kruskal(y_true, y_pred)
        kruskal_results.append({
            '模型': model_name,
            'H统计量': stat,
            'P值': p_value,
            '是否可靠 (P > 0.05)': '是' if p_value > 0.05 else '否'
        })

    kruskal_df = pd.DataFrame(kruskal_results)
    print(kruskal_df.to_string(index=False))

    # --- 2. Wilcoxon符号秩检验 ---
    print("\n\n--- 表5: Wilcoxon符号秩检验 (模型性能比较) ---")
    print("H0: 两个模型的预测误差相同。 P < 0.05 则拒绝H0，表明模型性能有显著差异。\n")

    wilcoxon_results = []
    # 计算每个模型的预测误差
    for result in detailed_results_list:
        result['error'] = np.abs(result['y_test_original'] - result['y_pred_test']).flatten()

    # 两两比较模型
    models = detailed_results_list
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1_name = models[i]['model_name']
            model2_name = models[j]['model_name']
            error1 = models[i]['error']
            error2 = models[j]['error']

            try:
                # 比较误差的差异，如果差异全为0，会报错
                if np.all(np.isclose(error1, error2)):  # 使用isclose处理浮点数比较
                    stat, p_value = 'N/A', 1.0
                else:
                    stat, p_value = stats.wilcoxon(error1, error2, alternative='two-sided')

                if p_value < 0.05:
                    if np.mean(error1) < np.mean(error2):
                        conclusion = f"{model1_name} 显著更优"
                    else:
                        conclusion = f"{model2_name} 显著更优"
                else:
                    conclusion = "无显著差异"

                wilcoxon_results.append({
                    '模型1': model1_name,
                    '模型2': model2_name,
                    'W统计量': stat,
                    'P值': p_value,
                    '结论': conclusion
                })
            except ValueError as e:
                wilcoxon_results.append({
                    '模型1': model1_name,
                    '模型2': model2_name,
                    'W统计量': 'N/A',
                    'P值': 'N/A',
                    '结论': f"检验出错: {e}"
                })

    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    print(wilcoxon_df.to_string(index=False))
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    run_experiment_and_tests()
    print("\n模型训练与统计检验已成功完成！")
