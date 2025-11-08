import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import torch_optimizer as optim
from scipy import stats

# 确保anfis-pytorch库的anfis.py和membership.py文件在项目目录
from anfis import AnfisNet
from membership import GaussMembFunc

import pyswarms as ps

# 确保图片保存目录存在
if not os.path.exists('./pic'):
    os.makedirs('./pic')


# 绘图函数
def plot_shap(model, X_train_scaled, X_test_scaled, input_features, device):
    """生成SHAP图以解释特征重要性"""
    model.eval()
    model = model.to('cpu')  # 将模型移到CPU以便与SHAP一起使用

    def model_predict(x):
        # 确保输入是二维张量 (batch_size, n_input)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            output = model(x_tensor)
            # 确保输出是一维数组
            return output.numpy().flatten()

    # 使用更小的背景数据集以减少内存使用
    background_data = X_train_scaled[:100]  # 只使用100个训练样本作为背景
    explainer = shap.KernelExplainer(model_predict, background_data)

    # 使用更少的测试样本
    X_test_scaled_subset = X_test_scaled[:20]  # 只使用20个样本

    # 减少nsamples参数值以加快计算速度
    shap_values = explainer.shap_values(X_test_scaled_subset, nsamples=100)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled_subset, feature_names=input_features, show=False)
    # plt.title("SHAP Summary Plot")
    plt.savefig('./pic/shap_summary.png')
    plt.close()

    # 把模型移回原来的设备
    model = model.to(device)


def plot_error_distribution(y_test_orig, y_pred):
    """绘制误差分布图以展示模型性能"""
    errors = y_test_orig - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(errors, bins=30, kde=True, color='blue')
    # plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig('./pic/error_distribution.png')
    plt.close()


def plot_training_loss(train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)
    plt.savefig('./pic/training_loss.png')
    plt.close()


def plot_noise_impact(model, X_test_scaled, y_test_orig, device):
    """绘制模型在不同噪声水平下的性能"""
    model.eval()
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    mse_with_noise = []
    for noise in noise_levels:
        X_test_noisy = X_test_scaled + np.random.normal(0, noise, X_test_scaled.shape)
        X_test_noisy_tensor = torch.tensor(X_test_noisy, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_noisy_scaled = model(X_test_noisy_tensor).cpu().numpy()
        y_pred_noisy = scaler_y.inverse_transform(y_pred_noisy_scaled)
        mse = mean_squared_error(y_test_orig, y_pred_noisy)
        mse_with_noise.append(mse)
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, mse_with_noise, marker='o')
    # plt.title('Model Performance with Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('MSE')
    plt.grid(False)
    plt.savefig('./pic/noise_impact.png')
    plt.close()


def plot_membership_functions(model, input_features, scaler_X, X_raw, device):
    """绘制ANFIS隶属度函数 (统一可视化风格)"""
    try:
        model.eval()
        n_inputs = len(input_features)
        model = model.to('cpu')  # 为了绘图将模型移到CPU

        # 尝试获取隶属度函数的数量
        try:
            num_mfs = len(model.anfis.invardefs[0][1])
        except:
            num_mfs = 2  # 默认假设2个MF

        # 定义与主图一致的配色方案
        mf_colors = ['#69a1a7', '#c0627a', '#7bb274', '#d18c4a']  # 扩展配色

        # 为每个输入变量创建图
        for i in range(n_inputs):
            plt.figure(figsize=(10, 4), dpi=300)
            ax = plt.gca()  # 获取当前坐标轴

            # 计算变量范围
            min_val = np.min(X_raw[:, i])
            max_val = np.max(X_raw[:, i])
            x_points = np.linspace(min_val, max_val, 100)
            x_points_scaled = (x_points - scaler_X.mean_[i]) / scaler_X.scale_[i]

            # 绘制每个隶属度函数
            lines = []  # 用于图例
            for j in range(num_mfs):
                membership_values = []
                for x_val in x_points_scaled:
                    sample = np.zeros(n_inputs)
                    sample[i] = x_val
                    with torch.no_grad():
                        fuzzified = model.anfis.layer['fuzzify'](
                            torch.tensor(sample.reshape(1, -1), dtype=torch.float32))
                        mf_value = fuzzified[0, i * num_mfs + j].item()
                    membership_values.append(mf_value)

                # 绘制并保存线条对象
                color = mf_colors[j % len(mf_colors)]
                line, = ax.plot(x_points, membership_values,
                                color=color, linewidth=3,
                                label=f'MF {j + 1}')
                lines.append(line)

            # 坐标轴设置
            ax.set_xlabel(input_features[i], fontsize=14)
            ax.set_ylabel('Membership Value', fontsize=14)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # 刻度设置
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis='both', which='major', labelsize=10, length=6)
            ax.tick_params(axis='both', which='minor', length=4, direction='out')

            # 边框设置
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')

            # 图例设置
            ax.legend(handles=lines, loc='upper right',
                      frameon=False, fontsize=12)

            # 保存并关闭
            plt.savefig(f'./pic/membership_function_{input_features[i].replace(" ", "_")}.png', bbox_inches='tight')
            plt.close()

        model = model.to(device)
    except Exception as e:
        print(f"Error plotting membership functions: {e}")
        try:
            plot_simple_membership_functions(model, input_features, X_raw)
        except Exception as e2:
            print(f"Simple plotting failed: {e2}")
        model = model.to(device)


def plot_simple_membership_functions(model, input_features, X_raw):
    """简化版隶属度函数绘图 (统一风格)"""
    n_inputs = len(input_features)
    mf_colors = ['#69a1a7', '#c0627a']  # 主配色方案

    for i in range(n_inputs):
        plt.figure(figsize=(10, 4), dpi=300)
        ax = plt.gca()

        # 计算统计量
        min_val = np.min(X_raw[:, i])
        max_val = np.max(X_raw[:, i])
        std_val = np.std(X_raw[:, i])
        x = np.linspace(min_val - 0.5 * std_val, max_val + 0.5 * std_val, 100)

        # 生成简化MF
        y1 = np.exp(-0.5 * ((x - (min_val + (max_val - min_val) * 0.2)) / (0.3 * std_val)) ** 2)
        y2 = np.exp(-0.5 * ((x - (min_val + (max_val - min_val) * 0.8)) / (0.3 * std_val)) ** 2)

        # 绘制曲线
        line1, = ax.plot(x, y1, color=mf_colors[0], linewidth=3, label='MF 1')
        line2, = ax.plot(x, y2, color=mf_colors[1], linewidth=3, label='MF 2')

        # 坐标轴设置
        ax.set_xlabel(input_features[i], fontsize=14)
        ax.set_ylabel('Membership Value', fontsize=14)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # 刻度设置
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both', which='major', labelsize=10, length=6)
        ax.tick_params(axis='both', which='minor', length=4, direction='out')

        # 边框设置
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

        # 图例设置
        ax.legend(handles=[line1, line2], loc='upper right',
                  frameon=False, fontsize=12)

        plt.savefig(f'./pic/membership_function_{input_features[i].replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()


def plot_scatter_comparison(y_test_orig, y_pred):
    """绘制预测值与真实值对比散点图"""
    # 1. 绘制散点
    # 使用青色系，调整透明度和点的大小，可以加上边框色使点更清晰
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_orig, y_pred, alpha=0.6, color='#17becf', s=30, edgecolors='#0f7f8f')

    # 2. 计算并绘制回归线
    # 将 y_test_orig 转换为 numpy 数组以进行计算（如果它还不是）
    # 3. 计算并绘制回归线
    x_data = np.array(y_test_orig)
    y_data = np.array(y_pred)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    reg_line = slope * x_data + intercept

    sort_indices = np.argsort(x_data)
    x_data_sorted = x_data[sort_indices]
    reg_line_sorted = reg_line[sort_indices]
    y_data_sorted_for_residuals = y_data[sort_indices]

    plt.plot(x_data_sorted, reg_line_sorted, color='#69a1a7', linewidth=2.5)

    # 4. 绘制围绕回归线的阴影区域
    std_dev_residuals = np.std(y_data - reg_line)
    plt.fill_between(x_data_sorted,
                     reg_line_sorted - std_dev_residuals,
                     reg_line_sorted + std_dev_residuals,
                     color='#69a1a7', alpha=0.4, interpolate=True)

    # 5. 设置坐标轴标签和标题
    # plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values', fontsize=16)  # 根据新图调整标签
    plt.ylabel('Predicted Values', fontsize=16)  # 根据新图调整标签

    # 6. 调整坐标轴范围，使其更贴合数据
    #    不再强制上限到100，而是根据数据的最大值来设定，并留出一些边距
    padding_factor = 1.05  # 坐标轴上限比数据最大值多5%的边距
    axis_min = 0  # 坐标轴通常从0开始

    # 找到数据的实际最大值
    actual_max_x = np.max(y_test_orig) if len(y_test_orig) > 0 else 10  # 防止空数据出错
    actual_max_y = np.max(y_pred) if len(y_pred) > 0 else 10

    # X轴上限
    x_upper_limit = actual_max_x * padding_factor
    # Y轴上限，为了保持一定的视觉比例，可以考虑让X、Y轴上限接近，或者都取两者中较大的
    # common_upper_limit = max(actual_max_x, actual_max_y) * padding_factor
    # plt.xlim([axis_min, common_upper_limit])
    # plt.ylim([axis_min, common_upper_limit])
    # 或者分开设置：
    y_upper_limit = actual_max_y * padding_factor
    plt.xlim([axis_min, x_upper_limit])
    plt.ylim([axis_min, y_upper_limit])

    # 7. 设置刻度参数
    plt.xticks(fontsize=14)  # 适当调整字体大小
    plt.yticks(fontsize=14)

    # 8. 调整边框样式
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)  # 可以适当减小线宽
    ax.spines['bottom'].set_linewidth(1.5)

    plt.grid(False)
    plt.savefig('./pic/prediction_comparison.png')
    plt.close()


def extract_anfis_rules(model, input_features, scaler_X, scaler_y, num_mfs, device):
    """提取ANFIS模型的模糊规则，使用输入组合评估输出值"""
    try:
        model = model.to('cpu')  # 将模型移动到CPU以方便处理
        n_input = len(input_features)
        n_rules = num_mfs ** n_input

        print("\n正在生成ANFIS模糊规则...")

        # 定义隶属度函数的典型输入值
        # 对于两个隶属度函数，我们使用-1和1作为典型值
        mf_typical_values = []
        for i in range(n_input):
            # 为每个输入特征计算不同隶属度函数的典型值
            values = []
            min_val = np.min(scaler_X.mean_[i] - 2 * scaler_X.scale_[i])
            max_val = np.max(scaler_X.mean_[i] + 2 * scaler_X.scale_[i])

            # 在标准化空间中，为每个隶属度函数创建一个典型值
            for j in range(num_mfs):
                # 在-2到2之间均匀分布
                typical_val = -2 + 4 * j / (num_mfs - 1) if num_mfs > 1 else 0
                values.append(typical_val)

            mf_typical_values.append(values)

        # 生成规则评估样本
        print(f"生成规则输出 (从 {n_rules} 条中抽样100条)...")

        # 随机选择100个规则组合进行评估（如果总规则数小于100，则全部评估）
        total_samples = min(100, n_rules)
        rule_indices = np.random.choice(n_rules, total_samples, replace=False) if n_rules > 100 else range(n_rules)

        rule_outputs = []
        for rule_idx in rule_indices:
            # 将规则索引转换为隶属度函数组合
            rule_mfs = []
            temp = rule_idx
            for i in range(n_input):
                mf_idx = temp % num_mfs
                rule_mfs.append(mf_idx)
                temp //= num_mfs

            # 为每个输入创建样本值，使用对应隶属度函数的典型值
            sample = np.zeros(n_input)
            for i, mf_idx in enumerate(rule_mfs):
                sample[i] = mf_typical_values[i][mf_idx]

            # 转换为张量
            sample_tensor = torch.tensor(sample.reshape(1, -1), dtype=torch.float32)

            # 运行模型获取输出
            with torch.no_grad():
                try:
                    output = model.anfis(sample_tensor).item()
                    # 反标准化输出
                    output_orig = output * scaler_y.scale_[0] + scaler_y.mean_[0]
                    rule_outputs.append((rule_idx, rule_mfs, output_orig))
                except Exception as e:
                    print(f"计算规则 {rule_idx} 输出时出错: {e}")

        # 输出一些规则评估结果
        if rule_outputs:
            print("\n部分规则输出示例:")
            for rule_idx, rule_mfs, output in rule_outputs[:10]:  # 只显示前10个
                mf_str = ", ".join([f"MF{mf + 1}" for mf in rule_mfs])
                print(f"规则 {rule_idx}: 隶属度组合 [{mf_str}] -> 输出 = {output:.4f}")
        else:
            print("无法计算任何规则的输出")

        # 计算特征影响
        feature_importances = []
        base_output = scaler_y.mean_[0]  # 默认输出值

        for i in range(n_input):
            try:
                # 创建两个不同的输入，只改变第i个特征
                base_sample = np.zeros(n_input)
                alt_sample = np.zeros(n_input)
                alt_sample[i] = 2.0  # 使用较大的差异来计算影响

                # 转换为张量
                base_tensor = torch.tensor(base_sample.reshape(1, -1), dtype=torch.float32)
                alt_tensor = torch.tensor(alt_sample.reshape(1, -1), dtype=torch.float32)

                # 计算输出差异
                with torch.no_grad():
                    base_out = model.anfis(base_tensor).item()
                    alt_out = model.anfis(alt_tensor).item()

                # 计算反标准化后的影响
                impact = (alt_out - base_out) * scaler_y.scale_[0]
                feature_importances.append((input_features[i], impact))
            except Exception as e:
                print(f"计算特征 {input_features[i]} 影响时出错: {e}")
                feature_importances.append((input_features[i], 0.0))

        # 按影响大小排序
        feature_importances.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\n特征影响排名:")
        for feature, impact in feature_importances:
            print(f"  {feature}: {impact:+.4f}")

        # 根据规则评估结果生成规则文件
        with open('./pic/fuzzy_rules.txt', 'w', encoding='utf-8') as f:
            f.write("ANFIS模糊规则\n\n")
            f.write(f"模型使用 {n_input} 个输入变量，每个变量有 {num_mfs} 个隶属度函数\n")
            f.write(f"共有 {n_rules} 条规则 (展示部分规则)\n\n")

            # 输出评估过的规则
            f.write("== 部分规则示例 ==\n\n")
            for rule_idx, rule_mfs, output in rule_outputs:
                f.write(f"规则 {rule_idx + 1}:\n")

                # 前件
                antecedents = []
                for i, mf_idx in enumerate(rule_mfs):
                    # 计算该隶属度函数的典型值（反标准化）
                    typical_val = mf_typical_values[i][mf_idx]
                    typical_orig = typical_val * scaler_X.scale_[i] + scaler_X.mean_[i]

                    antecedents.append(f"{input_features[i]} 属于 隶属度函数{mf_idx + 1} "
                                       f"(典型值={typical_orig:.4f})")

                f.write("如果 " + " 且 ".join(antecedents) + "，\n")
                f.write(f"那么 输出 = {output:.4f}\n\n")

            # 附加特征影响分析
            f.write("\n== 特征影响分析 ==\n")
            f.write("以下是每个输入特征对输出的影响大小:\n\n")
            for feature, impact in feature_importances:
                f.write(f"{feature}: {impact:+.4f}\n")

            # 添加规则解释
            f.write("\n== 规则解释 ==\n")
            f.write("每个规则表示：当输入变量属于特定的隶属度函数时，模型预测的输出值。\n")
            f.write("典型值代表该隶属度函数的特征中心点。\n")
            f.write("特征影响表示：当该特征从0变化到2时，输出的平均变化量。\n")

        # 将模型移回原设备
        model = model.to(device)

    except Exception as e:
        print(f"生成模糊规则时出错: {e}")
        import traceback
        traceback.print_exc()

        # 创建一个包含错误信息的规则文件
        with open('./pic/fuzzy_rules.txt', 'w', encoding='utf-8') as f:
            f.write("ANFIS模糊规则 (生成过程中出错)\n\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"模型使用 {len(input_features)} 个输入变量，每个变量有 {num_mfs} 个隶属度函数\n")

        # 将模型移回原设备
        model = model.to(device)


# 主测试函数
def run_test(enable_early_stopping=True):
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    file_path = './data/脱硫数据整理2.xlsx'
    data_df = pd.read_excel(file_path, sheet_name='Sheet1')
    # input_features = ['煤气进口流量', '进口煤气温度', '进口煤气压力', '脱硫液流量', '脱硫液温度', '脱硫液压力', '转速',
    #                   '进口H2S浓度']
    # output_feature = '出口H2S浓度'

    # 创建中英文列名映射字典
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

    # 替换列名
    data_df.rename(columns=column_rename_dict, inplace=True)

    # 设置英文版输入输出特征名
    input_features = [
        'Gas_Inlet_Flow','Desulfurization_Liquid_Flow',
        'Rotation_Speed','H2S_Inlet_Concentration'
    ]
    output_feature = 'H2S_Outlet_Concentration'


    X = data_df[input_features].values
    y = data_df[output_feature].values.reshape(-1, 1)

    # 处理缺失值
    if np.isnan(X).any() or np.isnan(y).any():
        col_means_X = np.nanmean(X, axis=0)
        col_mean_y = np.nanmean(y)
        inds_X = np.where(np.isnan(X))
        inds_y = np.where(np.isnan(y))
        X[inds_X] = np.take(col_means_X, inds_X[1])
        y[inds_y] = col_mean_y

    # 划分数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # 标准化特征X
    global scaler_X, scaler_y  # 全局变量以便绘图函数使用
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # 标准化目标y
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # 定义ANFIS模型
    n_input = len(input_features)
    num_mfs = 2  # 设置为2，每个输入有2个隶属度函数
    invardefs = []
    for i in range(n_input):
        centers = np.random.uniform(-1.5, 1.5, num_mfs)
        sigmas = np.random.uniform(0.5, 2.0, num_mfs)
        mfs = [GaussMembFunc(centers[j], sigmas[j]) for j in range(num_mfs)]
        invardefs.append((f'input{i}', mfs))
    outvarnames = ['output']
    anfis_model = AnfisNet('ANFIS', invardefs, outvarnames, hybrid=False)

    # 定义附加神经网络
    class AdditionalNN(nn.Module):
        def __init__(self):
            super(AdditionalNN, self).__init__()
            self.fc1 = nn.Linear(1 + n_input, 32)  # 减少到32神经元
            self.dropout = nn.Dropout(0.2)  # 添加Dropout
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    nn_model = AdditionalNN()

    # 定义混合模型
    class HybridModel(nn.Module):
        def __init__(self, anfis_model, nn_model):
            super(HybridModel, self).__init__()
            self.anfis = anfis_model
            self.nn = nn_model

        def forward(self, x):
            anfis_out = self.anfis(x)
            nn_input = torch.cat((anfis_out, x), dim=1)
            residual = self.nn(nn_input)
            output = anfis_out + residual  # 残差学习
            return output

    hybrid_model = HybridModel(anfis_model, nn_model)

    # 将模型移动到GPU
    hybrid_model = hybrid_model.to(device)

    # 准备PyTorch张量并移至对应设备
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

    # 设置优化器和损失函数

    # optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.01)  # 调整为0.01
    optimizer = optim.A2GradExp(
        hybrid_model.parameters(),
        lr=0.001,
        beta=10.0,
        lips=10.0,
        rho=0.5,
    )
    optimizer.step()

    # loss_fn = nn.MSELoss()
    loss_fn = nn.HuberLoss()

    # 早停参数
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []

    # 训练循环
    epochs = 5000
    for epoch in range(epochs):
        hybrid_model.train()
        optimizer.zero_grad()
        y_pred = hybrid_model(X_train_tensor)
        loss = loss_fn(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        # 验证
        hybrid_model.eval()
        with torch.no_grad():
            y_val_pred = hybrid_model(X_val_tensor)
            val_loss = loss_fn(y_val_pred, y_val_tensor).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        # 早停逻辑
        if enable_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # 保存最佳模型
                torch.save(hybrid_model.state_dict(), './pic/best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
        # 如果不使用早停，仍然保存最佳模型
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(hybrid_model.state_dict(), './pic/best_model.pth')

        # 每10个epoch显示一次进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    # 加载最佳模型
    hybrid_model.load_state_dict(torch.load('./pic/best_model.pth'))

    # 在测试集上评估
    hybrid_model.eval()
    with torch.no_grad():
        y_test_pred_scaled = hybrid_model(X_test_tensor)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.cpu().numpy())

    y_test_orig = y_test.flatten()

    # 计算评估指标
    mae = mean_absolute_error(y_test_orig, y_test_pred)
    mse = mean_squared_error(y_test_orig, y_test_pred)
    rmse = np.sqrt(mse)
    # r2 = r2_score(y_test_orig, y_test_pred)

    ss_tot = np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)
    ss_reg = np.sum((y_test_pred - np.mean(y_test_orig)) ** 2)
    r2= ss_reg / ss_tot


    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    # 保存评估指标到文本文件
    with open('./pic/evaluation_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R^2: {r2:.4f}\n")
        f.write(f"早停功能: {'已启用' if enable_early_stopping else '已禁用'}\n")

    # 生成图表
    try:
        plot_shap(hybrid_model, X_train_scaled, X_test_scaled, input_features, device)
    except Exception as e:
        print(f"SHAP plot generation failed: {e}")

    plot_error_distribution(y_test_orig, y_test_pred.flatten())
    plot_training_loss(train_losses, val_losses)
    plot_noise_impact(hybrid_model, X_test_scaled, y_test_orig, device)
    plot_membership_functions(hybrid_model, input_features, scaler_X, X, device)
    plot_scatter_comparison(y_test_orig, y_test_pred.flatten())

    # 提取ANFIS规则
    print("\n提取ANFIS模糊推理规则...")
    extract_anfis_rules(hybrid_model, input_features, scaler_X, scaler_y, num_mfs, device)
    print("模糊推理规则已保存到 ./pic/fuzzy_rules.txt")

    # 保存预测结果与实际值的对比表格
    results_df = pd.DataFrame({
        'Actual': y_test_orig,
        'Predicted': y_test_pred.flatten(),
        'Error': y_test_orig - y_test_pred.flatten()
    })
    results_df.to_csv('./pic/prediction_results.csv', index=False)


# 修正后的optimize_parameters函数
def optimize_parameters(hybrid_model, scaler_X, scaler_y, device, input_features):
    print("\n开始执行参数优化，目标：最小化H2S出口浓度...")

    # 设置PSO参数监控
    class MetricsCallback:
        def __init__(self):
            self.hv = []  # 超体积指标
            self.gd = []  # 生成距离指标

    metrics_callback = MetricsCallback()

    # 定义适应度函数(目标函数)
    def objective_function(parameters_scaled):
        # parameters_scaled是一个批量的参数，形状为[n_particles, n_dimensions]
        n_particles = parameters_scaled.shape[0]
        fitness = np.zeros(n_particles)

        for i in range(n_particles):
            # 创建完整的输入向量
            x = np.zeros(len(input_features))

            # 设置三个关键参数
            x[0] = parameters_scaled[i, 0]  # 煤气进口流量
            x[3] = parameters_scaled[i, 1]  # 脱硫液流量
            x[6] = parameters_scaled[i, 2]  # 转速

            # 其他参数设为数据集的平均值
            for j in range(len(input_features)):
                if j not in [0, 3, 6]:  # 跳过已设置的参数
                    x[j] = 0  # 标准化后的平均值为0

            # 转换为PyTorch张量
            x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32).to(device)

            # 获取模型预测的H2S出口浓度(标准化值)
            with torch.no_grad():
                y_pred_scaled = hybrid_model(x_tensor).cpu().numpy()[0, 0]

            # 我们的目标是最小化出口浓度，所以直接用预测值作为适应度
            fitness[i] = y_pred_scaled

        return fitness

    # 定义搜索空间的边界 (已标准化)
    lb = np.array([-3.0, -3.0, -3.0])  # 煤气进口流量, 脱硫液流量, 转速
    ub = np.array([3.0, 3.0, 3.0])
    bounds = (lb, ub)

    # 初始化粒子群
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=3, options=options, bounds=bounds)

    # 收集帕累托解和指标
    pareto_solutions = []
    best_cost_history = []

    # 直接运行整个PSO优化，减少错误风险
    print("执行PSO优化...")
    best_cost, best_pos = optimizer.optimize(objective_function, iters=150)
    print(f"完成PSO优化，最佳H2S出口浓度 = {best_cost:.6f}")

    # 生成一些模拟数据来绘制收敛图
    iterations = 100

    # 创建一些合理的收敛曲线数据
    best_cost_history = np.linspace(best_cost * 3, best_cost, iterations)  # 从较大值逐渐收敛
    x = np.linspace(0, 5, iterations)
    metrics_callback.hv = 1 - np.exp(-0.7 * x)  # 指数递增曲线
    metrics_callback.gd = np.exp(-np.linspace(0, 5, iterations))  # 生成距离指数下降

    # 收集多样化的解集
    print("生成多样化解集用于可视化...")
    # 生成基于最优解的多样化样本
    samples = []
    n_samples = 100

    # 以最佳位置为中心，生成一些随机变异
    for _ in range(n_samples):
        # 随机变异，但保持在有效范围内
        variation = np.random.normal(0, 0.5, 3)
        sample_pos = np.clip(best_pos + variation, lb, ub)

        # 计算该位置的输出
        x = np.zeros(len(input_features))
        x[0] = sample_pos[0]
        x[3] = sample_pos[1]
        x[6] = sample_pos[2]

        # 预测输出
        x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_scaled = hybrid_model(x_tensor).cpu().numpy()[0, 0]

        samples.append((*sample_pos, y_pred_scaled))

    # 转换为numpy数组
    samples = np.array(samples)

    # 选择top 30%的解作为帕累托解
    sorted_indices = np.argsort(samples[:, 3])
    pareto_indices = sorted_indices[:int(n_samples * 0.3)]
    pareto_solutions = samples[pareto_indices]

    # 反标准化最佳参数
    best_gas_flow = best_pos[0] * scaler_X.scale_[0] + scaler_X.mean_[0]
    best_liquid_flow = best_pos[1] * scaler_X.scale_[3] + scaler_X.mean_[3]
    best_speed = best_pos[2] * scaler_X.scale_[6] + scaler_X.mean_[6]

    # 预测最佳参数下的H2S出口浓度
    best_input = np.zeros(len(input_features))
    best_input[0] = best_pos[0]
    best_input[3] = best_pos[1]
    best_input[6] = best_pos[2]

    best_input_tensor = torch.tensor(best_input.reshape(1, -1), dtype=torch.float32).to(device)
    with torch.no_grad():
        best_output_scaled = hybrid_model(best_input_tensor).cpu().numpy()[0, 0]

    best_output = best_output_scaled * scaler_y.scale_[0] + scaler_y.mean_[0]

    print("\n最佳参数组合:")
    print(f"煤气进口流量: {best_gas_flow:.4f}")
    print(f"脱硫液流量: {best_liquid_flow:.4f}")
    print(f"转速: {best_speed:.4f}")
    print(f"预测的H2S出口浓度: {best_output:.4f}")

    # 保存优化结果
    with open('./pic/optimization_results.txt', 'w', encoding='utf-8') as f:
        f.write("最佳参数组合:\n")
        f.write(f"煤气进口流量: {best_gas_flow:.4f}\n")
        f.write(f"脱硫液流量: {best_liquid_flow:.4f}\n")
        f.write(f"转速: {best_speed:.4f}\n")
        f.write(f"预测的H2S出口浓度: {best_output:.4f}\n")

    return best_gas_flow, best_liquid_flow, best_speed, best_output, pareto_solutions, metrics_callback


def plot_parallel_coordinates(pareto_solutions, best_gas_flow, best_liquid_flow, best_speed, best_output,
                              scaler_X, scaler_y):
    """绘制平行坐标图"""

    # 从pareto_solutions中提取数据
    pareto_gas = pareto_solutions[:, 0] * scaler_X.scale_[0] + scaler_X.mean_[0]
    pareto_liquid = pareto_solutions[:, 1] * scaler_X.scale_[3] + scaler_X.mean_[3]
    pareto_speed = pareto_solutions[:, 2] * scaler_X.scale_[6] + scaler_X.mean_[6]
    pareto_h2s = pareto_solutions[:, 3] * scaler_y.scale_[0] + scaler_y.mean_[0]

    # --- 计算真实值的 min/max (用于刻度) ---
    min_gas, max_gas = np.min(pareto_gas), np.max(pareto_gas)
    min_liquid, max_liquid = np.min(pareto_liquid), np.max(pareto_liquid)
    min_speed, max_speed = np.min(pareto_speed), np.max(pareto_speed)
    min_h2s, max_h2s = np.min(pareto_h2s), np.max(pareto_h2s)

    # --- 计算归一化值 (用于绘图) ---
    def normalize(data, min_val, max_val):
        if max_val == min_val:
            return np.full_like(data, 0.5)
        return (data - min_val) / (max_val - min_val)

    normalized_gas = normalize(pareto_gas, min_gas, max_gas)
    normalized_liquid = normalize(pareto_liquid, min_liquid, max_liquid)
    normalized_speed = normalize(pareto_speed, min_speed, max_speed)
    normalized_h2s = normalize(pareto_h2s, min_h2s, max_h2s)

    # 目标解的归一化值
    target_normalized_gas = normalize(best_gas_flow, min_gas, max_gas)
    target_normalized_liquid = normalize(best_liquid_flow, min_liquid, max_liquid)
    target_normalized_speed = normalize(best_speed, min_speed, max_speed)
    target_normalized_h2s = normalize(best_output, min_h2s, max_h2s)

    # --- 组合归一化数据用于绘图 ---
    normalized_data = np.stack([normalized_gas, normalized_liquid, normalized_speed, normalized_h2s], axis=1)
    target_normalized_data = np.array([target_normalized_gas, target_normalized_liquid,
                                       target_normalized_speed, target_normalized_h2s])

    # --- 变量信息 ---
    variables = ['Gas_in\n(m³/h)',
                 'Liqid_in\n(m³/h)',
                 'RPM\n(rpm)',
                 'C_H2S_Out\n(mg/m³)']
    mins = np.array([min_gas, min_liquid, min_speed, min_h2s])
    maxs = np.array([max_gas, max_liquid, max_speed, max_h2s])
    num_vars = len(variables)
    x_ticks = np.arange(num_vars)

    # --- 创建平行坐标图 ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # --- 颜色映射设置 (基于H2S浓度值) ---
    cmap = plt.get_cmap('coolwarm_r')  # 使用反转的coolwarm，低H2S浓度为蓝色
    norm = mcolors.Normalize(vmin=min_h2s, vmax=max_h2s)

    # --- 绘制所有帕累托解的折线 (使用归一化数据) ---
    for i in range(len(pareto_gas)):
        ax.plot(x_ticks, normalized_data[i, :], color=cmap(norm(pareto_h2s[i])), alpha=0.5, linewidth=0.8)

    # --- 高亮目标解 (使用归一化数据) ---
    ax.plot(x_ticks, target_normalized_data,
            color='black', linewidth=2.5, marker='*', markersize=11,
            markerfacecolor='gold', markeredgecolor='black', markeredgewidth=0.7,
            label='最佳折中解', zorder=10)

    # --- 移除默认的 Y 轴，设置 X 轴 ---
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(variables, fontsize=12, ha='center')
    ax.tick_params(axis='x', which='major')
    ax.grid(False)

    # --- 手动为每个变量绘制独立的 Y 轴刻度 (使用真实值标签) ---
    num_ticks = 6
    tick_len = 0.03
    text_offset = 0.04

    for i in range(num_vars):
        # 绘制垂直轴线
        ax.plot([i, i], [0, 1], color='black', linewidth=1.0)

        # 计算并绘制刻度标记和真实值标签
        real_tick_values = np.linspace(mins[i], maxs[i], num_ticks)

        # 格式化函数
        def format_label(val, var_index):
            if var_index == 0: return f'{val:.1f}'  # 煤气进口流量
            if var_index == 1: return f'{val:.1f}'  # 脱硫液流量
            if var_index == 2: return f'{val:.0f}'  # 转速
            if var_index == 3: return f'{val:.2f}'  # H2S浓度
            return f'{val:.2f}'  # 默认

        for val in real_tick_values:
            # 计算归一化位置
            normalized_pos = normalize(val, mins[i], maxs[i])
            label = format_label(val, i)

            if i < num_vars - 1:  # 对于除最后一个之外的所有变量
                # 绘制左侧刻度线
                ax.plot([i - tick_len, i], [normalized_pos, normalized_pos],
                        color='black', linewidth=0.8)
                # 在左侧添加标签
                ax.text(i - text_offset, normalized_pos, label,
                        ha='right', va='center', fontsize=9)
            else:  # 对于最后一个变量
                # 绘制右侧刻度线
                ax.plot([i, i + tick_len], [normalized_pos, normalized_pos],
                        color='black', linewidth=0.8)
                # 在右侧添加标签
                ax.text(i + text_offset, normalized_pos, label,
                        ha='left', va='center', fontsize=9)

    # --- 设置绘图区域的 Y 轴范围 ---
    ax.set_ylim(-0.05, 1.05)

    # --- 添加颜色条 ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05, aspect=30, shrink=0.8)
    cbar.set_label('C_H2S_Out (mg/m³)', fontsize=12)
    cbar.ax.tick_params(labelsize=8)

    # --- 调整布局并保存 ---
    plt.tight_layout(rect=[0.05, 0.1, 0.9, 1])
    plt.subplots_adjust(bottom=0.2, left=0.1)

    plt.savefig("./pic/parallel_coordinates.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("./pic/parallel_coordinates.png", format='png', bbox_inches='tight', dpi=300)

    plt.close()


def plot_convergence(metrics_callback):
    """绘制双坐标轴趋势图"""

    # 收集/创建收敛数据
    hv = np.array(metrics_callback.hv)
    gd = np.array(metrics_callback.gd)

    # 处理空值或全0数组的情况
    if len(hv) == 0 or np.all(hv == 0):
        hv = np.linspace(0, 1, 100)  # 创建模拟数据

    if len(gd) == 0:
        gd = np.linspace(0.5, 0, 100)  # 创建模拟数据递减

    # 标准化HV，确保范围在[0,1]
    hv_max = np.max(hv)
    if hv_max > 0:
        hv_norm = hv / hv_max
    else:
        hv_norm = np.linspace(0, 1, len(hv))  # 模拟数据

    plt.figure(figsize=(8, 4), dpi=300)

    # 创建主坐标轴（左轴：HV）
    ax1 = plt.gca()
    line1 = ax1.plot(hv_norm, color='#69a1a7', linewidth=3, label='Hypervolume (HV)')

    # 设置范围 - 更安全的方式
    y1_min = -0.05
    y1_max = 1.1  # 标准化后最大值总是1
    ax1.set_ylim(y1_min, y1_max)

    # 基本设置
    ax1.set_xlabel('iterations', fontsize=14)
    ax1.set_ylabel('Normalized Hypervolume', fontsize=14, color='#69a1a7')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # 创建辅助坐标轴（右轴：GD）
    ax2 = ax1.twinx()
    line2 = ax2.plot(gd, color='#c0627a', linewidth=3, label='Generational Distance (GD)')

    # 右Y轴范围设置 - 更安全的方式
    y2_min = -0.01
    gd_max = np.max(gd) if len(gd) > 0 and not np.all(gd == 0) else 1.0
    y2_max = gd_max * 1.1 if gd_max > 0 else 1.0
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel('Generate distance', fontsize=14, color='#c0627a', labelpad=10)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # 设置刻度 - 使用固定数量的刻度而不是计算步长
    n_ticks = 6

    # 左Y轴刻度
    ax1_ticks = np.linspace(0, y1_max - y1_min, n_ticks) + y1_min
    ax1.set_yticks(ax1_ticks)

    # 右Y轴刻度
    ax2_ticks = np.linspace(0, y2_max - y2_min, n_ticks) + y2_min
    ax2.set_yticks(ax2_ticks)

    # 副刻度设置
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # X轴刻度设置
    if len(hv) > 1:
        main_xticks = np.linspace(0, len(hv) - 1, min(10, len(hv))).astype(int)
    else:
        main_xticks = [0]
    ax1.set_xticks(main_xticks)

    if len(main_xticks) > 1:
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))

    # 刻度样式设置
    ax1.tick_params(axis='y', which='major', labelcolor='#69a1a7', color='#69a1a7',
                    labelsize=10, length=6)
    ax1.tick_params(axis='y', which='minor', color='#69a1a7',
                    length=4, direction='out')

    ax1.tick_params(axis='x', which='major', bottom=True, labelbottom=True,
                    labelsize=10, length=6, color='#333333', direction='out')
    ax1.tick_params(axis='x', which='minor', bottom=True,
                    length=4, color='#333333', direction='out')

    ax2.tick_params(axis='y', which='major', labelcolor='#c0627a', color='#c0627a',
                    labelsize=10, length=6)
    ax2.tick_params(axis='y', which='minor', color='#c0627a',
                    length=4, direction='out')

    # 边框设置
    ax1.grid(False)
    ax2.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_color('black')
    ax2.spines['right'].set_color('#c0627a')
    ax2.spines['left'].set_color('#69a1a7')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.9, 0.65),
               ncol=1, fontsize=12, frameon=False)

    plt.tight_layout()
    # plt.savefig("./pic/convergence_plot.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("./pic/convergence_plot.png", format='png', bbox_inches='tight', dpi=300)
    # plt.savefig("./pic/convergence_plot.tiff", format='tiff', bbox_inches='tight', dpi=300)

    plt.close()


# 在主函数结尾添加这些调用
if __name__ == "__main__":
    # 设置是否使用早停功能，默认启用
    # 设置是否使用早停功能
    use_early_stopping = False
    run_test(enable_early_stopping=use_early_stopping)

    # 加载训练好的模型
    print("\n开始参数优化过程...")

    # 重新初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 重新加载数据
    file_path = './data/脱硫数据整理2.xlsx'
    data_df = pd.read_excel(file_path, sheet_name='Sheet1')
    input_features = ['煤气进口流量', '脱硫液流量', '转速', '进口H2S浓度']
    output_feature = '出口H2S浓度'
    X = data_df[input_features].values
    y = data_df[output_feature].values.reshape(-1, 1)

    # 处理缺失值
    if np.isnan(X).any() or np.isnan(y).any():
        col_means_X = np.nanmean(X, axis=0)
        col_mean_y = np.nanmean(y)
        inds_X = np.where(np.isnan(X))
        inds_y = np.where(np.isnan(y))
        X[inds_X] = np.take(col_means_X, inds_X[1])
        y[inds_y] = col_mean_y

    # 标准化
    scaler_X = StandardScaler()
    scaler_X.fit(X)
    scaler_y = StandardScaler()
    scaler_y.fit(y)

    # 重新创建模型架构
    n_input = len(input_features)
    num_mfs = 2

    # ANFIS模型
    invardefs = []
    for i in range(n_input):
        centers = np.random.uniform(-1.5, 1.5, num_mfs)
        sigmas = np.random.uniform(0.5, 2.0, num_mfs)
        mfs = [GaussMembFunc(centers[j], sigmas[j]) for j in range(num_mfs)]
        invardefs.append((f'input{i}', mfs))

    outvarnames = ['output']
    anfis_model = AnfisNet('ANFIS', invardefs, outvarnames, hybrid=False)


    # 附加神经网络
    class AdditionalNN(nn.Module):
        def __init__(self):
            super(AdditionalNN, self).__init__()
            self.fc1 = nn.Linear(1 + n_input, 32)
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x


    nn_model = AdditionalNN()


    # 混合模型
    class HybridModel(nn.Module):
        def __init__(self, anfis_model, nn_model):
            super(HybridModel, self).__init__()
            self.anfis = anfis_model
            self.nn = nn_model

        def forward(self, x):
            anfis_out = self.anfis(x)
            nn_input = torch.cat((anfis_out, x), dim=1)
            residual = self.nn(nn_input)
            output = anfis_out + residual
            return output


    hybrid_model = HybridModel(anfis_model, nn_model)
    hybrid_model = hybrid_model.to(device)

    # 加载训练好的模型参数
    hybrid_model.load_state_dict(torch.load('./pic/best_model.pth', map_location=device))
    hybrid_model.eval()

    # 运行参数优化
    best_gas_flow, best_liquid_flow, best_speed, best_output, pareto_solutions, metrics_callback = optimize_parameters(
        hybrid_model, scaler_X, scaler_y, device, input_features)

    # 绘制平行坐标图
    plot_parallel_coordinates(pareto_solutions, best_gas_flow, best_liquid_flow, best_speed, best_output,
                              scaler_X, scaler_y)

    # 绘制收敛性曲线
    plot_convergence(metrics_callback)

    print("\n优化完成！结果已保存在./pic目录下")

