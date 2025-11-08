import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sanfis import SANFIS, plottingtools
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 在训练完成后添加以下可视化代码
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 加载数据
file_path = './data/脱硫数据整理2.xlsx'
try:
    data_df = pd.read_excel(file_path, sheet_name='Sheet1')
except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径：{file_path}")
    exit()

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

# 输入特征
input_features = [
    'Gas_Inlet_Flow',
    'Desulfurization_Liquid_Flow',
    'Rotation_Speed',
    'H2S_Inlet_Concentration'
]
output_feature = 'H2S_Outlet_Concentration'

# 检查数据是否为空和是否有缺失值
if data_df.empty:
    print("错误：加载的数据为空。")
    exit()

# 检查缺失值
print("数据缺失值检查：")
print(data_df[input_features + [output_feature]].isnull().sum())

# 删除包含缺失值的行
data_clean = data_df[input_features + [output_feature]].dropna()
print(f"清理后数据量：{len(data_clean)} 行")

X = data_clean[input_features].values
y = data_clean[output_feature].values.reshape(-1, 1)

# 检查数据范围
print(f"输入特征X的范围：")
for i, feature in enumerate(input_features):
    print(f"  {feature}: min={X[:, i].min():.4f}, max={X[:, i].max():.4f}")
print(f"输出y的范围: min={y.min():.4f}, max={y.max():.4f}")

# 数据标准化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

# SANFIS需要状态输入S
S_train_tensor = X_train_tensor
S_test_tensor = X_test_tensor

# 准备训练数据
train_data = [S_train_tensor, X_train_tensor, y_train_tensor]
valid_data = [S_test_tensor, X_test_tensor, y_test_tensor]

# --- 改进的SANFIS模型参数配置 ---
n_input_features = X_scaled.shape[1]
n_memb_funcs_per_input = 3  # 增加隶属函数数量到3个

# 构建隶属函数配置
membfuncs_config = []

for i in range(n_input_features):
    # 为每个输入特征生成多个隶属函数的均值
    mu_values = np.linspace(0.1, 0.9, n_memb_funcs_per_input).tolist()
    sigma_values = [0.2] * n_memb_funcs_per_input  # 适当的sigma值

    membfuncs_config.append({
        'function': 'gaussian',
        'n_memb': n_memb_funcs_per_input,
        'params': {
            'mu': {'value': mu_values, 'trainable': True},
            'sigma': {'value': sigma_values, 'trainable': True}
        }
    })

print(f"隶属函数配置：")
for i, config in enumerate(membfuncs_config):
    print(f"  输入{i + 1}: {config['params']['mu']['value']}")

# 初始化SANFIS模型
model = SANFIS(
    membfuncs=membfuncs_config,
    n_input=n_input_features,
    to_device='cpu',
    scale='Std'
)

# 检查模型参数
print(f"\n模型参数数量：")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总可训练参数：{total_params}")

# 定义损失函数和优化器
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化器，调高学习率

print(f"\n开始训练SANFIS模型...")

# 手动训练循环以监控损失
epochs = 2000
train_losses = []
valid_losses = []

model.train()
for epoch in range(epochs):
    # 前向传播
    optimizer.zero_grad()

    # 训练集预测
    train_pred = model(train_data[0], train_data[1])  # S, X
    train_loss = loss_function(train_pred, train_data[2])  # y

    # 反向传播
    train_loss.backward()

    # 检查梯度
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # 验证集评估
    model.eval()
    with torch.no_grad():
        valid_pred = model(valid_data[0], valid_data[1])
        valid_loss = loss_function(valid_pred, valid_data[2])
    model.train()

    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss.item())

    # 每10个epoch打印一次
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss.item():.6f}, "
              f"Valid Loss: {valid_loss.item():.6f}, "
              f"Grad Norm: {grad_norm:.6f}")

print(f"SANFIS模型训练完成。")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(valid_losses, label='Validation Loss', alpha=0.7)
plt.title('SANFIS Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 测试集预测
model.eval()
with torch.no_grad():
    y_pred_scaled_tensor = model(S_test_tensor, X_test_tensor)

# 转换为numpy并反标准化
y_pred_scaled = y_pred_scaled_tensor.detach().numpy()
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_np)

# 计算误差
mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print(f"\n模型评估结果 (基于测试集):")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")
print(f"最终训练损失: {train_losses[-1]:.6f}")
print(f"最终验证损失: {valid_losses[-1]:.6f}")

# 绘制预测结果对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred, alpha=0.6)
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title(f'SANFIS预测结果 (R² = {r2:.4f})')
plt.grid(True, alpha=0.3)
plt.show()


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_membership_functions(model, input_features, n_points=100):
    """可视化隶属度函数"""
    n_inputs = len(input_features)

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    # 为每个输入特征绘制隶属函数
    for i in range(n_inputs):
        ax = axes[i]

        # 获取该输入的隶属函数参数
        # SANFIS模型中隶属函数参数存储在model.membfuncs中
        try:
            # 提取mu和sigma参数
            mu_params = []
            sigma_params = []

            # 遍历模型的参数来找到隶属函数参数
            for name, param in model.named_parameters():
                if f'membfunc_{i}' in name or f'input_{i}' in name:
                    if 'mu' in name or 'mean' in name:
                        mu_params.extend(param.detach().cpu().numpy().tolist())
                    elif 'sigma' in name or 'std' in name:
                        sigma_params.extend(param.detach().cpu().numpy().tolist())

            # 如果直接提取参数失败，使用配置的初始值
            if not mu_params or not sigma_params:
                mu_params = membfuncs_config[i]['params']['mu']['value']
                sigma_params = membfuncs_config[i]['params']['sigma']['value']

            # 创建x轴范围（标准化后的范围[0,1]）
            x = np.linspace(0, 1, n_points)

            # 绘制每个隶属函数
            for j, (mu, sigma) in enumerate(zip(mu_params, sigma_params)):
                # 高斯隶属函数
                membership = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                ax.plot(x, membership, linewidth=2,
                        label=f'MF{j + 1} (μ={mu:.3f}, σ={sigma:.3f})')

            ax.set_title(f'{input_features[i]}的隶属函数', fontsize=12, fontweight='bold')
            ax.set_xlabel('标准化输入值')
            ax.set_ylabel('隶属度')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1.1)

        except Exception as e:
            print(f"绘制输入{i}的隶属函数时出错: {e}")
            ax.text(0.5, 0.5, f'无法提取参数\n{input_features[i]}',
                    ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()


def extract_fuzzy_rules(model, input_features):
    """提取并可视化模糊规则"""
    print("\n=== 模糊规则分析 ===")

    # 计算规则数量
    n_memb_funcs_per_input = n_memb_funcs_per_input_global = 3
    n_inputs = len(input_features)
    total_rules = n_memb_funcs_per_input ** n_inputs

    print(f"输入变量数: {n_inputs}")
    print(f"每个输入的隶属函数数: {n_memb_funcs_per_input}")
    print(f"总规则数: {total_rules}")

    # 生成所有可能的规则组合
    from itertools import product
    rule_combinations = list(product(range(n_memb_funcs_per_input), repeat=n_inputs))

    # 使用一些测试数据来评估规则权重
    test_samples = 10
    X_sample = X_train_tensor[:test_samples]
    S_sample = S_train_tensor[:test_samples]

    model.eval()
    with torch.no_grad():
        # 获取隶属度值
        # 这里需要根据SANFIS的具体实现来获取中间计算结果
        pred = model(S_sample, X_sample)

    return rule_combinations


def visualize_rule_weights(model, input_features, n_samples=50):
    """可视化规则权重和激活强度"""

    # 使用训练数据的一个子集来分析规则激活
    X_sample = X_train_tensor[:n_samples]
    S_sample = S_train_tensor[:n_samples]

    model.eval()

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 计算每个样本的预测和规则激活
    activations = []
    predictions = []

    with torch.no_grad():
        for i in range(n_samples):
            x_single = X_sample[i:i + 1]
            s_single = S_sample[i:i + 1]
            pred = model(s_single, x_single)
            predictions.append(pred.item())

            # 计算隶属度（简化版本）
            sample_activation = []
            for j in range(len(input_features)):
                # 模拟隶属度计算
                x_val = x_single[0, j].item()
                for k in range(n_memb_funcs_per_input):
                    mu = membfuncs_config[j]['params']['mu']['value'][k]
                    sigma = membfuncs_config[j]['params']['sigma']['value'][k]
                    membership = np.exp(-0.5 * ((x_val - mu) / sigma) ** 2)
                    sample_activation.append(membership)
            activations.append(sample_activation)

    activations = np.array(activations)

    # 绘制规则激活热力图
    im1 = ax1.imshow(activations.T, cmap='viridis', aspect='auto')
    ax1.set_title('规则激活强度热力图', fontweight='bold')
    ax1.set_xlabel('样本编号')
    ax1.set_ylabel('隶属函数编号')
    plt.colorbar(im1, ax=ax1, label='激活强度')

    # 绘制预测值分布
    ax2.hist(predictions, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_title('模型预测值分布', fontweight='bold')
    ax2.set_xlabel('预测值')
    ax2.set_ylabel('频次')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印激活统计
    print(f"\n=== 规则激活统计 ===")
    print(f"平均激活强度: {np.mean(activations):.4f}")
    print(f"最大激活强度: {np.max(activations):.4f}")
    print(f"最小激活强度: {np.min(activations):.4f}")


def analyze_input_importance(model, input_features):
    """分析输入特征重要性"""
    print(f"\n=== 输入特征重要性分析 ===")

    # 计算每个输入特征的参数方差作为重要性指标
    importance_scores = []

    for i, feature in enumerate(input_features):
        # 获取该特征对应的隶属函数参数
        try:
            mu_params = membfuncs_config[i]['params']['mu']['value']
            sigma_params = membfuncs_config[i]['params']['sigma']['value']

            # 计算参数的变异程度作为重要性指标
            mu_variance = np.var(mu_params)
            sigma_mean = np.mean(sigma_params)
            importance = mu_variance / (sigma_mean + 1e-8)
            importance_scores.append(importance)

        except:
            importance_scores.append(0)

    # 标准化重要性分数
    importance_scores = np.array(importance_scores)
    if np.sum(importance_scores) > 0:
        importance_scores = importance_scores / np.sum(importance_scores)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    bars = plt.bar(input_features, importance_scores, color='skyblue', edgecolor='navy')
    plt.title('输入特征重要性分析', fontweight='bold', fontsize=14)
    plt.xlabel('输入特征')
    plt.ylabel('重要性分数')
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar, score in zip(bars, importance_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 打印重要性排序
    feature_importance = list(zip(input_features, importance_scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print("特征重要性排序:")
    for i, (feature, score) in enumerate(feature_importance):
        print(f"{i + 1}. {feature}: {score:.4f}")


# 主可视化函数
def visualize_sanfis_model(model, input_features):
    """完整的SANFIS模型可视化"""
    print("开始SANFIS模型可视化...")

    # 1. 可视化隶属函数
    print("\n1. 可视化隶属函数...")
    visualize_membership_functions(model, input_features)

    # 2. 提取并分析模糊规则
    print("\n2. 分析模糊规则...")
    rule_combinations = extract_fuzzy_rules(model, input_features)

    # 3. 可视化规则权重和激活
    print("\n3. 可视化规则激活...")
    visualize_rule_weights(model, input_features)

    # 4. 分析输入重要性
    print("\n4. 分析输入特征重要性...")
    analyze_input_importance(model, input_features)


# 在训练完成后调用可视化
print(f"\n{'=' * 50}")
print(f"开始SANFIS模型结构可视化")
print(f"{'=' * 50}")

# 执行所有可视化
visualize_sanfis_model(model, input_features)


# 额外：模型参数总结
def print_model_summary(model, input_features):
    """打印模型参数总结"""
    print(f"\n=== SANFIS模型参数总结 ===")
    print(f"输入特征数: {len(input_features)}")
    print(f"每个输入的隶属函数数: {n_memb_funcs_per_input}")
    print(f"总规则数: {n_memb_funcs_per_input ** len(input_features)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数数: {total_params}")
    print(f"可训练参数数: {trainable_params}")

    print(f"\n模型层结构:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            print(f"  {name}: {module}")


print_model_summary(model, input_features)
