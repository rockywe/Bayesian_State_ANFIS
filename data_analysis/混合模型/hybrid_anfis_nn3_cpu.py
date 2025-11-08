import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import sys
import argparse
from itertools import product

# 确保anfis-pytorch库的anfis.py和membership.py文件在项目目录
from anfis import AnfisNet
from membership import GaussMembFunc, BellMembFunc

# 解析命令行参数
parser = argparse.ArgumentParser(description='ANFIS-NN混合模型训练')
parser.add_argument('--early_stop', type=int, default=0, help='是否使用早停 (1=是, 0=否)')
parser.add_argument('--patience', type=int, default=20, help='早停的耐心值')
parser.add_argument('--epochs', type=int, default=500, help='最大训练轮数')
parser.add_argument('--lr', type=float, default=0.005, help='学习率')
parser.add_argument('--num_mfs', type=int, default=2, help='每个输入的隶属度函数数量')
parser.add_argument('--hidden_size', type=int, default=64, help='神经网络隐藏层大小')
parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
args = parser.parse_args()

# 确保图片保存目录存在
if not os.path.exists('./pic'):
    os.makedirs('./pic')


# 绘图函数
def plot_shap(model, X_train_scaled, X_test_scaled, input_features):
    """生成SHAP图以解释特征重要性"""
    model.eval()

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
    plt.title("SHAP Summary Plot")
    plt.savefig('./pic/shap_summary.png')
    plt.close()


def plot_error_distribution(y_test_orig, y_pred):
    """绘制误差分布图以展示模型性能"""
    errors = y_test_orig - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(errors, bins=30, kde=True, color='blue')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig('./pic/error_distribution.png')
    plt.close()


def plot_training_loss(train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./pic/training_loss.png')
    plt.close()


def plot_noise_impact(model, X_test_scaled, y_test_orig):
    """绘制模型在不同噪声水平下的性能"""
    model.eval()
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    mse_with_noise = []
    for noise in noise_levels:
        X_test_noisy = X_test_scaled + np.random.normal(0, noise, X_test_scaled.shape)
        X_test_noisy_tensor = torch.tensor(X_test_noisy, dtype=torch.float32)
        with torch.no_grad():
            y_pred_noisy_scaled = model(X_test_noisy_tensor).numpy()
        y_pred_noisy = scaler_y.inverse_transform(y_pred_noisy_scaled)
        mse = mean_squared_error(y_test_orig, y_pred_noisy)
        mse_with_noise.append(mse)
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, mse_with_noise, marker='o')
    plt.title('Model Performance with Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig('./pic/noise_impact.png')
    plt.close()


def plot_membership_functions(model, input_features, scaler_X, X_raw):
    """绘制ANFIS隶属度函数"""
    try:
        model.eval()
        n_inputs = len(input_features)

        # 尝试获取隶属度函数的数量
        try:
            num_mfs = len(model.anfis.invardefs[0][1])
        except:
            # 如果上面的方法不起作用，使用参数值
            num_mfs = args.num_mfs

        print(f"检测到 {num_mfs} 个隶属度函数")

        # 为每个输入变量创建一个图
        for i in range(n_inputs):
            plt.figure(figsize=(10, 4))

            # 计算该变量的范围
            min_val = np.min(X_raw[:, i])
            max_val = np.max(X_raw[:, i])

            # 创建范围内的100个点进行可视化
            x_points = np.linspace(min_val, max_val, 100)

            # 对这些点进行标准化处理
            x_points_scaled = (x_points - scaler_X.mean_[i]) / scaler_X.scale_[i]

            # 为每个隶属度函数计算值
            for j in range(num_mfs):
                membership_values = []

                for x_val in x_points_scaled:
                    # 创建样本张量，全部设为0
                    sample = np.zeros(n_inputs)
                    sample[i] = x_val
                    sample_tensor = torch.tensor(sample.reshape(1, -1), dtype=torch.float32)

                    # 获取隶属度值
                    with torch.no_grad():
                        fuzzified = model.anfis.layer['fuzzify'](sample_tensor)
                        # 隶属度的索引取决于ANFIS的实现，这里假设按照顺序排列
                        mf_value = fuzzified[0, i * num_mfs + j].item()

                    membership_values.append(mf_value)

                # 绘制隶属度函数
                plt.plot(x_points, membership_values, label=f'MF {j + 1}')

            plt.title(f'Membership Functions for {input_features[i]}')
            plt.xlabel(input_features[i])
            plt.ylabel('Membership Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'./pic/membership_function_{input_features[i].replace(" ", "_")}.png')
            plt.close()
    except Exception as e:
        print(f"Error plotting membership functions: {e}")
        # 提供简化版的绘图方法
        try:
            plot_simple_membership_functions(model, input_features, X_raw)
        except Exception as e2:
            print(f"Simple membership function plotting also failed: {e2}")


def plot_simple_membership_functions(model, input_features, X_raw):
    """简化版的隶属度函数绘图"""
    n_inputs = len(input_features)
    num_mfs = args.num_mfs  # 使用命令行参数

    # 对每个输入变量创建输入范围的可视化
    for i in range(n_inputs):
        plt.figure(figsize=(10, 4))

        # 计算该变量的一些统计数据
        min_val = np.min(X_raw[:, i])
        max_val = np.max(X_raw[:, i])
        mean_val = np.mean(X_raw[:, i])
        std_val = np.std(X_raw[:, i])

        # 创建用于可视化的x轴范围
        x = np.linspace(min_val - 0.5 * std_val, max_val + 0.5 * std_val, 100)

        # 创建num_mfs个高斯隶属度函数作为简化表示
        colors = ['r-', 'g-', 'b-', 'c-', 'm-', 'y-', 'k-']
        for j in range(num_mfs):
            center = min_val + j * (max_val - min_val) / (num_mfs - 1) if num_mfs > 1 else mean_val
            y = np.exp(-0.5 * ((x - center) / (0.3 * std_val)) ** 2)
            plt.plot(x, y, colors[j % len(colors)], label=f'MF {j + 1}')

        plt.title(f'Simplified Membership Functions for {input_features[i]}')
        plt.xlabel(input_features[i])
        plt.ylabel('Membership Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./pic/membership_function_{input_features[i].replace(" ", "_")}.png')
        plt.close()


def plot_scatter_comparison(y_test_orig, y_pred):
    """绘制预测值与真实值对比散点图"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred, alpha=0.6)

    # 添加理想预测线（y=x）
    min_val = min(np.min(y_test_orig), np.min(y_pred))
    max_val = max(np.max(y_test_orig), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # 添加评估指标到图中
    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred)

    plt.annotate(f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig('./pic/prediction_comparison.png')
    plt.close()


def extract_anfis_rules(model, input_features, scaler_X, scaler_y, num_mfs):
    """提取ANFIS模型的模糊规则，使用输入组合评估输出值"""
    try:
        n_input = len(input_features)
        n_rules = num_mfs ** n_input

        print("\n正在生成ANFIS模糊规则...")

        # 定义隶属度函数的典型输入值
        mf_typical_values = []
        for i in range(n_input):
            # 为每个输入特征计算不同隶属度函数的典型值
            values = []
            for j in range(num_mfs):
                # 在-2到2之间均匀分布
                typical_val = -2 + 4 * j / (num_mfs - 1) if num_mfs > 1 else 0
                values.append(typical_val)

            mf_typical_values.append(values)

        # 生成规则评估样本
        print(f"生成规则输出 (从 {n_rules} 条中抽样最多100条)...")

        # 随机选择100个规则组合进行评估（如果总规则数小于100，则全部评估）
        max_samples = 100
        total_samples = min(max_samples, n_rules)
        rule_indices = np.random.choice(n_rules, total_samples, replace=False) if n_rules > max_samples else range(
            n_rules)

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

    except Exception as e:
        print(f"生成模糊规则时出错: {e}")
        import traceback
        traceback.print_exc()

        # 创建一个包含错误信息的规则文件
        with open('./pic/fuzzy_rules.txt', 'w', encoding='utf-8') as f:
            f.write("ANFIS模糊规则 (生成过程中出错)\n\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"模型使用 {len(input_features)} 个输入变量，每个变量有 {num_mfs} 个隶属度函数\n")


# 特征工程函数
def engineer_features(X, input_features):
    """创建额外的特征"""
    print("执行特征工程...")
    X_new = X.copy()
    new_features = []

    # 添加比率特征
    n_features = X.shape[1]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # 避免除以零，添加一个小的常数
            ratio = X[:, i] / (X[:, j] + 1e-6)
            X_new = np.column_stack((X_new, ratio))
            new_features.append(f"{input_features[i]}/{input_features[j]}")

    # 添加2阶交互项
    for i in range(n_features):
        for j in range(i, n_features):
            interaction = X[:, i] * X[:, j]
            X_new = np.column_stack((X_new, interaction))
            if i == j:
                new_features.append(f"{input_features[i]}^2")
            else:
                new_features.append(f"{input_features[i]}*{input_features[j]}")

    # 添加平方根变换
    for i in range(n_features):
        # 确保值非负
        min_val = np.min(X[:, i])
        if min_val < 0:
            transformed = np.sqrt(X[:, i] - min_val + 1e-6)
        else:
            transformed = np.sqrt(X[:, i] + 1e-6)
        X_new = np.column_stack((X_new, transformed))
        new_features.append(f"sqrt({input_features[i]})")

    # 添加对数变换
    for i in range(n_features):
        # 确保值为正
        min_val = np.min(X[:, i])
        if min_val <= 0:
            transformed = np.log(X[:, i] - min_val + 1 + 1e-6)
        else:
            transformed = np.log(X[:, i] + 1e-6)
        X_new = np.column_stack((X_new, transformed))
        new_features.append(f"log({input_features[i]})")

    extended_features = input_features + new_features
    print(f"特征工程后的特征数量: {len(extended_features)}")

    return X_new, extended_features


# 数据清洗函数
def clean_data(X, y, contamination=0.05):
    """使用IsolationForest检测并移除异常值"""
    print("检测异常值...")
    # 组合X和y以同时应用异常值检测
    combined = np.hstack((X, y))

    # 使用IsolationForest检测异常值
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    yhat = iso_forest.fit_predict(combined)
    mask = yhat != -1  # -1表示异常

    # 应用掩码来过滤数据
    X_clean = X[mask]
    y_clean = y[mask]

    print(f"移除了 {X.shape[0] - X_clean.shape[0]} 个异常值，保留了 {X_clean.shape[0]} 个样本")
    return X_clean, y_clean


# 主测试函数
def run_test():
    # 加载数据
    file_path = './data/脱硫数据整理.xlsx'
    data_df = pd.read_excel(file_path, sheet_name='Sheet1')
    input_features = ['煤气进口流量', '进口煤气温度', '进口煤气压力', '脱硫液流量', '脱硫液温度', '脱硫液压力', '转速',
                      '进口H2S浓度']
    output_feature = '出口H2S浓度'
    X = data_df[input_features].values
    y = data_df[output_feature].values.reshape(-1, 1)
    # 为全局作用域提供最佳模型的变量
    global scaler_X, scaler_y

    # 处理缺失值
    if np.isnan(X).any() or np.isnan(y).any():
        col_means_X = np.nanmean(X, axis=0)
        col_mean_y = np.nanmean(y)
        inds_X = np.where(np.isnan(X))
        inds_y = np.where(np.isnan(y))
        X[inds_X] = np.take(col_means_X, inds_X[1])
        y[inds_y] = col_mean_y

    # 清洗数据，移除异常值
    X, y = clean_data(X, y, contamination=0.05)

    # 特征工程
    X_eng, extended_features = engineer_features(X, input_features)

    # 创建交叉验证折
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 记录每折的性能指标
    fold_metrics = []

    # 记录最佳模型
    best_model = None
    best_metrics = {"r2": -float('inf')}

    # 交叉验证
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_eng)):
        print(f"\n===== 开始第 {fold + 1}/{n_splits} 折训练 =====")

        X_train_val, X_test = X_eng[train_idx], X_eng[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

        # 标准化特征X - 使用RobustScaler以更好地处理异常值
        global scaler_X, scaler_y  # 全局变量以便绘图函数使用
        scaler_X = RobustScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        # 标准化目标y
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)

        # 定义ANFIS模型
        n_input = X_train.shape[1]
        num_mfs = args.num_mfs
        invardefs = []
        for i in range(n_input):
            # 随机选择隶属度函数类型（高斯或钟形）
            if np.random.random() < 0.5:  # 50% 高斯，50% 钟形
                centers = np.random.uniform(-1.5, 1.5, num_mfs)
                sigmas = np.random.uniform(0.5, 2.0, num_mfs)
                mfs = [GaussMembFunc(centers[j], sigmas[j]) for j in range(num_mfs)]
            else:
                a_values = np.random.uniform(0.5, 2.0, num_mfs)
                b_values = np.random.uniform(1.0, 3.0, num_mfs)
                c_values = np.random.uniform(-1.5, 1.5, num_mfs)
                mfs = [BellMembFunc(a_values[j], b_values[j], c_values[j]) for j in range(num_mfs)]
            invardefs.append((f'input{i}', mfs))
        outvarnames = ['output']
        anfis_model = AnfisNet('ANFIS', invardefs, outvarnames, hybrid=True)  # 启用hybrid=True

        # 定义改进的附加神经网络
        class AdditionalNN(nn.Module):
            def __init__(self, n_input, hidden_size=args.hidden_size, dropout_rate=0.3):
                super(AdditionalNN, self).__init__()
                self.fc1 = nn.Linear(1 + n_input, hidden_size)
                self.bn1 = nn.BatchNorm1d(hidden_size)  # 添加批归一化
                self.dropout1 = nn.Dropout(dropout_rate)

                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.bn2 = nn.BatchNorm1d(hidden_size // 2)
                self.dropout2 = nn.Dropout(dropout_rate)

                self.fc3 = nn.Linear(hidden_size // 2, 1)

            def forward(self, x):
                x = F.leaky_relu(self.bn1(self.fc1(x)))  # 使用LeakyReLU激活
                x = self.dropout1(x)

                x = F.leaky_relu(self.bn2(self.fc2(x)))
                x = self.dropout2(x)

                x = self.fc3(x)
                return x

        nn_model = AdditionalNN(n_input)

        # 定义混合模型
        class HybridModel(nn.Module):
            def __init__(self, anfis_model, nn_model, alpha=0.7):
                super(HybridModel, self).__init__()
                self.anfis = anfis_model
                self.nn = nn_model
                self.alpha = nn.Parameter(torch.tensor([alpha]), requires_grad=True)  # 可学习参数

            def forward(self, x):
                anfis_out = self.anfis(x)
                nn_input = torch.cat((anfis_out, x), dim=1)
                residual = self.nn(nn_input)
                # 使用sigmoid确保alpha在0-1之间
                alpha = torch.sigmoid(self.alpha)
                output = alpha * anfis_out + (1 - alpha) * residual  # 加权组合
                return output

        hybrid_model = HybridModel(anfis_model, nn_model)

        # 准备PyTorch张量
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


        # 设置优化器和损失函数
        optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=args.lr, weight_decay=1e-4)

        # 创建自定义损失函数，增加大误差惩罚
        def weighted_mse_loss(inputs, targets):
            mse = (inputs - targets) ** 2
            weighted = mse * (1 + torch.abs(inputs - targets))  # 给大误差更大的惩罚
            return torch.mean(weighted)

        loss_fn = weighted_mse_loss

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

        # 早停参数
        patience = args.patience if args.early_stop else float('inf')
        best_val_loss = float('inf')
        counter = 0
        train_losses = []
        val_losses = []

        # 训练循环
        epochs = args.epochs
        for epoch in range(epochs):
            hybrid_model.train()
            epoch_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = hybrid_model(batch_X)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()

                # 梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(train_dataset)

            # 验证
            hybrid_model.eval()
            with torch.no_grad():
                y_val_pred = hybrid_model(X_val_tensor)
                val_loss = loss_fn(y_val_pred, y_val_tensor).item()

            train_losses.append(epoch_loss)
            val_losses.append(val_loss)

            # 调整学习率
            scheduler.step(val_loss)

            # 早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # 保存本折最佳模型
                fold_best_state = hybrid_model.state_dict()
            else:
                counter += 1
                if args.early_stop and counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

            # 每20个epoch显示一次进度
            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 加载本折最佳模型
        hybrid_model.load_state_dict(fold_best_state)

        # 在测试集上评估
        hybrid_model.eval()
        with torch.no_grad():
            y_test_pred_scaled = hybrid_model(X_test_tensor)
            y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.numpy())

        y_test_orig = y_test.flatten()

        # 计算评估指标
        mae = mean_absolute_error(y_test_orig, y_test_pred)
        mse = mean_squared_error(y_test_orig, y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_orig, y_test_pred)

        # 记录本折指标
        fold_metrics.append({
            "fold": fold + 1,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        })

        print(f"\n第 {fold + 1} 折结果:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R^2: {r2:.4f}")

        # 保存最佳模型
        if r2 > best_metrics["r2"]:
            best_metrics = {
                "fold": fold + 1,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2
            }
            best_model = hybrid_model
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_scaler_X = scaler_X
            best_scaler_y = scaler_y
            best_y_test = y_test_orig
            best_y_pred = y_test_pred
            best_X_test_scaled = X_test_scaled
            best_X_raw = X_test

    # 计算平均性能指标
    avg_metrics = {
        "mae": np.mean([m["mae"] for m in fold_metrics]),
        "mse": np.mean([m["mse"] for m in fold_metrics]),
        "rmse": np.mean([m["rmse"] for m in fold_metrics]),
        "r2": np.mean([m["r2"] for m in fold_metrics])
    }

    # 输出交叉验证的总结
    print("\n===== 交叉验证总结 =====")
    for fold, metrics in enumerate(fold_metrics):
        print(f"第 {fold + 1} 折: MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}, "
              f"RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

    print("\n平均性能:")
    print(f"平均 MAE: {avg_metrics['mae']:.4f}")
    print(f"平均 MSE: {avg_metrics['mse']:.4f}")
    print(f"平均 RMSE: {avg_metrics['rmse']:.4f}")
    print(f"平均 R²: {avg_metrics['r2']:.4f}")

    print("\n最佳模型 (第 {fold} 折):")
    print(f"最佳 MAE: {best_metrics['mae']:.4f}")
    print(f"最佳 MSE: {best_metrics['mse']:.4f}")
    print(f"最佳 RMSE: {best_metrics['rmse']:.4f}")
    print(f"最佳 R²: {best_metrics['r2']:.4f}")


    scaler_X = best_scaler_X
    scaler_y = best_scaler_y

    # 保存最佳模型
    torch.save(best_model.state_dict(), './pic/best_model.pth')

    # 保存评估指标到文本文件
    with open('./pic/evaluation_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("===== 交叉验证结果 =====\n\n")
        for fold, metrics in enumerate(fold_metrics):
            f.write(f"第 {fold + 1} 折: MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}\n")

        f.write("\n===== 平均性能 =====\n")
        f.write(f"平均 MAE: {avg_metrics['mae']:.4f}\n")
        f.write(f"平均 MSE: {avg_metrics['mse']:.4f}\n")
        f.write(f"平均 RMSE: {avg_metrics['rmse']:.4f}\n")
        f.write(f"平均 R²: {avg_metrics['r2']:.4f}\n")

        f.write("\n===== 最佳模型 (第 {fold} 折) =====\n")
        f.write(f"最佳 MAE: {best_metrics['mae']:.4f}\n")
        f.write(f"最佳 MSE: {best_metrics['mse']:.4f}\n")
        f.write(f"最佳 RMSE: {best_metrics['rmse']:.4f}\n")
        f.write(f"最佳 R²: {best_metrics['r2']:.4f}\n")

        f.write("\n===== 模型配置 =====\n")
        f.write(f"隶属度函数数量: {args.num_mfs}\n")
        f.write(f"隐藏层大小: {args.hidden_size}\n")
        f.write(f"批大小: {args.batch_size}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"使用早停: {'是' if args.early_stop else '否'}\n")
        if args.early_stop:
            f.write(f"早停耐心值: {args.patience}\n")
        f.write(f"最大轮数: {args.epochs}\n")

    # 生成图表
    try:
        plot_shap(best_model, X_train_scaled, best_X_test_scaled, extended_features)
    except Exception as e:
        print(f"SHAP plot generation failed: {e}")

    plot_error_distribution(best_y_test, best_y_pred.flatten())
    plot_training_loss(best_train_losses, best_val_losses)
    plot_noise_impact(best_model, best_X_test_scaled, best_y_test)

    # 使用原始数据X来绘制隶属度函数
    try:
        plot_membership_functions(best_model, extended_features, best_scaler_X, best_X_raw)
    except Exception as e:
        print(f"Membership function plotting failed: {e}")
        plot_simple_membership_functions(best_model, extended_features, best_X_raw)

    plot_scatter_comparison(best_y_test, best_y_pred.flatten())

    # 提取ANFIS规则
    print("\n提取ANFIS模糊推理规则...")
    extract_anfis_rules(best_model, extended_features, best_scaler_X, best_scaler_y, args.num_mfs)
    print("模糊推理规则已保存到 ./pic/fuzzy_rules.txt")

    # 保存预测结果与实际值的对比表格
    results_df = pd.DataFrame({
        'Actual': best_y_test,
        'Predicted': best_y_pred.flatten(),
        'Error': best_y_test - best_y_pred.flatten()
    })
    results_df.to_csv('./pic/prediction_results.csv', index=False)

    return best_metrics


if __name__ == "__main__":
    # 如果没有命令行参数，提供默认值
    if len(sys.argv) == 1:
        print("使用默认参数运行...")

    print(f"参数设置:\n"
          f"  早停: {'启用' if args.early_stop else '禁用'}\n"
          f"  耐心值: {args.patience}\n"
          f"  最大轮数: {args.epochs}\n"
          f"  学习率: {args.lr}\n"
          f"  隶属度函数数: {args.num_mfs}\n"
          f"  隐藏层大小: {args.hidden_size}\n"
          f"  批大小: {args.batch_size}")

    # 运行测试
    best_metrics = run_test()

    print("\n训练完成！")
    print(f"最佳模型性能 - R²: {best_metrics['r2']:.4f}, RMSE: {best_metrics['rmse']:.4f}")
