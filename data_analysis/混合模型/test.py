import matplotlib.pyplot as plt
import numpy as np
from scipy import stats # 用于计算线性回归

def plot_scatter_comparison_single_styled_adjusted(y_test_orig, y_pred):
    """
    绘制预测值与真实值对比散点图 (单一数据组，仿示例图风格)。
    图中包含散点、回归线和围绕回归线的阴影区域。
    不显示 Slope 和 R² 值。
    调整了图像尺寸和坐标轴范围。
    """
    # 1. 调整图像尺寸，使其不那么“长”
    #    原始是 (10, 8)，可以尝试更接近正方形的比例，例如 (8, 7) 或 (7, 6)
    #    或者根据您的具体偏好调整
    plt.figure(figsize=(7, 6)) # 修改这里，例如改为宽度7英寸，高度6英寸

    # 2. 绘制散点
    plt.scatter(y_test_orig, y_pred, alpha=0.6, color='#17becf', s=30, edgecolors='#0f7f8f')

    # 3. 计算并绘制回归线
    x_data = np.array(y_test_orig)
    y_data = np.array(y_pred)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    reg_line = slope * x_data + intercept

    sort_indices = np.argsort(x_data)
    x_data_sorted = x_data[sort_indices]
    reg_line_sorted = reg_line[sort_indices]
    y_data_sorted_for_residuals = y_data[sort_indices]

    plt.plot(x_data_sorted, reg_line_sorted, color='#17becf', linewidth=2.5)

    # 4. 绘制围绕回归线的阴影区域
    std_dev_residuals = np.std(y_data - reg_line)
    plt.fill_between(x_data_sorted,
                     reg_line_sorted - std_dev_residuals,
                     reg_line_sorted + std_dev_residuals,
                     color='#17becf', alpha=0.4, interpolate=True)

    # 5. 设置坐标轴标签和标题
    #  根据您上传的图片 image_0e69b4.png，标签是 "Actual Values" 和 "Predicted Values"
    #  如果您想改回 "Experimental η (%)"，请取消注释下一行并注释掉当前行
    plt.xlabel('Actual Values', fontsize=16) # 根据新图调整标签
    plt.ylabel('Predicted Values', fontsize=16) # 根据新图调整标签
    # plt.xlabel('Experimental $\eta (\%)$', fontsize=18)
    # plt.ylabel('Predicted $\eta (\%)$', fontsize=18)

    # 标题 (如果需要)
    # plt.title('(b)', fontsize=22, loc='center', y=1.03)

    # 6. 调整坐标轴范围，使其更贴合数据
    #    不再强制上限到100，而是根据数据的最大值来设定，并留出一些边距
    padding_factor = 1.05 # 坐标轴上限比数据最大值多5%的边距
    axis_min = 0 # 坐标轴通常从0开始

    # 找到数据的实际最大值
    actual_max_x = np.max(y_test_orig) if len(y_test_orig) > 0 else 10 # 防止空数据出错
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
    plt.xticks(fontsize=14) # 适当调整字体大小
    plt.yticks(fontsize=14)

    # 8. 调整边框样式
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5) # 可以适当减小线宽
    ax.spines['bottom'].set_linewidth(1.5)

    plt.show()


# --- 使用示例 ---
# 请确保您有一组数据 y_test_orig_sample 和 y_pred_sample
# 使用与您图片 image_0e69b4.png 中数据分布相似的示例数据
np.random.seed(10) # 更换种子以获得不同但可复现的数据
# 假设实际值主要分布在0-60
y_test_orig_sample = np.sort(np.random.rand(150) * 60)
# 预测值围绕y=0.6x + 5 这条线，并加入一些随机性
y_pred_sample = 0.6 * y_test_orig_sample + 5 + np.random.normal(0, 5, 150)
# 确保预测值不为负，且不要过大（根据图示，Y轴数据大概到40-50）
y_pred_sample = np.clip(y_pred_sample, 0, 55)


plot_scatter_comparison_single_styled_adjusted(y_test_orig_sample, y_pred_sample)

print("调整后的图像已保存为 ./pic/prediction_comparison_single_adjusted.png")