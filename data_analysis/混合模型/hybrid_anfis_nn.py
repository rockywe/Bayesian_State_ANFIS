import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# 确保anfis-pytorch库的anfis.py和membership.py文件在项目目录
from anfis import AnfisNet
from membership import GaussMembFunc

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
        num_mfs = 3  # 假设每个输入有3个隶属度函数

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

        # 创建3个高斯隶属度函数作为简化表示
        y1 = np.exp(-0.5 * ((x - (min_val + (max_val - min_val) * 0.2)) / (0.3 * std_val)) ** 2)
        y2 = np.exp(-0.5 * ((x - mean_val) / (0.3 * std_val)) ** 2)
        y3 = np.exp(-0.5 * ((x - (min_val + (max_val - min_val) * 0.8)) / (0.3 * std_val)) ** 2)

        plt.plot(x, y1, 'r-', label='MF 1')
        plt.plot(x, y2, 'g-', label='MF 2')
        plt.plot(x, y3, 'b-', label='MF 3')

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

    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig('./pic/prediction_comparison.png')
    plt.close()


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
    num_mfs = 3  # 增加到3个隶属函数
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

    # 准备PyTorch张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.01)  # 调整为0.01
    loss_fn = nn.MSELoss()

    # 早停参数
    patience = 100
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []

    # 训练循环
    epochs = 20
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

        # 每10个epoch显示一次进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    # 加载最佳模型
    hybrid_model.load_state_dict(torch.load('./pic/best_model.pth'))

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

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")  # 使用 R^2 代替 R² 以避免编码问题

    # 保存评估指标到文本文件
    with open('./pic/evaluation_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R^2: {r2:.4f}\n")  # 使用 R^2 代替 R² 以避免编码问题

    # 生成图表
    try:
        plot_shap(hybrid_model, X_train_scaled, X_test_scaled, input_features)
    except Exception as e:
        print(f"SHAP plot generation failed: {e}")

    plot_error_distribution(y_test_orig, y_test_pred.flatten())
    plot_training_loss(train_losses, val_losses)
    plot_noise_impact(hybrid_model, X_test_scaled, y_test_orig)

    # 使用原始数据X来绘制隶属度函数
    plot_membership_functions(hybrid_model, input_features, scaler_X, X)
    plot_scatter_comparison(y_test_orig, y_test_pred.flatten())

    # 保存预测结果与实际值的对比表格
    results_df = pd.DataFrame({
        'Actual': y_test_orig,
        'Predicted': y_test_pred.flatten(),
        'Error': y_test_orig - y_test_pred.flatten()
    })
    results_df.to_csv('./pic/prediction_results.csv', index=False)


if __name__ == "__main__":
    run_test()
