import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 定义三角隶属函数
def triangular_membership_function(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# 模拟理论模型的值
def simulate_theoretical_values():
    return np.array([0.6010, 0.6923, 0.7799, 0.8534, 0.8876, 0.9078, 0.9212, 0.9309, 0.9382, 0.9439]), \
           np.array([0.5834, 0.6737, 0.7631, 0.8401, 0.8766, 0.8984, 0.9130, 0.9236, 0.9316, 0.9379]), \
           np.array([0.4918, 0.5659, 0.6547, 0.7454, 0.7942, 0.8256, 0.8478, 0.8645, 0.8775, 0.8880])

# 实际去除效率
def actual_removal_efficiency():
    return np.array([0.5976, 0.8068, 0.8996, 0.9222, 0.9882, 0.9918, 0.9932, 0.994, 0.993, 0.9922]), \
           np.array([0.4642, 0.6564, 0.8052, 0.9198, 0.9074, 0.9116, 0.9148, 0.9106, 0.9078, 0.9056]), \
           np.array([0.3332, 0.4426, 0.4976, 0.5312, 0.5666, 0.5922, 0.5788, 0.5832, 0.565, 0.5634])

# 计算误差
def calculate_error(simulated, actual):
    return 0.0005 * (1 - actual) - 0.0005 * (1 - simulated)

# 计算隶属度函数
def calculate_membership_functions(pH_values, gravity_values):
    y1_pH = triangular_membership_function(pH_values, [11.0, 12.0, 13.0])
    y2_pH = triangular_membership_function(pH_values, [12.0, 13.0, 14.0])
    y1_gravity = triangular_membership_function(gravity_values, [0, 70, 100])
    y2_gravity = triangular_membership_function(gravity_values, [100, 120, 200])
    return y1_pH, y2_pH, y1_gravity, y2_gravity

# 计算模糊规则权重
def calculate_rule_weights(y1_pH, y2_pH, y1_gravity, y2_gravity):
    rule_1 = y1_pH * y1_gravity
    rule_2 = y1_pH * y2_gravity
    rule_3 = y2_pH * y1_gravity
    rule_4 = y2_pH * y2_gravity

    rule_sum = rule_1 + rule_2 + rule_3 + rule_4

    beta_1 = rule_1 / rule_sum
    beta_2 = rule_2 / rule_sum
    beta_3 = rule_3 / rule_sum
    beta_4 = rule_4 / rule_sum

    return beta_1, beta_2, beta_3, beta_4

# 构建设计矩阵 Y
def build_design_matrix(beta_values, pH_values, gravity_values):
    beta_pH = [beta * pH_values for beta in beta_values]
    beta_gravity = [beta * gravity_values for beta in beta_values]
    return np.column_stack(list(beta_values) + beta_pH + beta_gravity)

# 计算插值
def perform_interpolation(pph, y_remo, pH_range):
    interpolated_results = np.zeros((len(pH_range), y_remo.shape[0]))
    for i in range(y_remo.shape[0]):
        cs = CubicSpline(pph, y_remo[i, :])
        interpolated_results[:, i] = cs(pH_range)
    return interpolated_results

# 生成网格数据
def generate_grid(pH_range, gravity_range):
    pH_grid, gravity_grid = np.meshgrid(pH_range, gravity_range)
    return pH_grid, gravity_grid

# 计算隶属度函数
def calculate_membership_functions_on_grid(pH_grid, gravity_grid):
    pH_low_membership = triangular_membership_function(pH_grid, [11.0, 12.0, 13.0])
    pH_high_membership = triangular_membership_function(pH_grid, [12.0, 13.0, 14.0])

    gravity_low_membership = triangular_membership_function(gravity_grid, [0, 70, 100])
    gravity_high_membership = triangular_membership_function(gravity_grid, [100, 120, 200])

    return pH_low_membership, pH_high_membership, gravity_low_membership, gravity_high_membership

# 绘制3D表面图
def plot_3d_surface(pH_range, gravity_range, removal_efficiency):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    pH_grid, gravity_grid = np.meshgrid(pH_range, gravity_range)
    ax.plot_surface(pH_grid, gravity_grid, removal_efficiency, cmap='viridis')
    ax.set_xlabel('pH', fontsize=12)
    ax.set_ylabel('Gravity Level', fontsize=12)
    ax.set_zlabel('NO removal efficiency (%)', fontsize=12)
    ax.set_title('Effect of pH and Gravity Level on Removal Efficiency', fontsize=14)
    plt.show()

# 替换数组中的0值
def replace_zeros(array, epsilon=1e-4):
    return np.where(array == 0, epsilon, array)

# def replace_zeros(matrix, epsilon=1e-4):
#     matrix[matrix == 0] = epsilon
#     return matrix
def main():
    # 模拟和实际数据
    yno_1, yno_2, yno_3 = simulate_theoretical_values()
    y_135, y_125, y_115 = actual_removal_efficiency()

    # 计算误差
    er1 = calculate_error(yno_1, y_135)
    er2 = calculate_error(yno_2, y_125)
    er3 = calculate_error(yno_3, y_115)

    # 输入参数
    pH1 = np.array([13.5] * 10 + [12.5] * 10 + [11.5] * 10)
    gravity_level = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200] * 3)
    error = np.concatenate([er1, er2, er3])

    # 计算隶属度函数
    y1_pH, y2_pH, y1_gravity, y2_gravity = calculate_membership_functions(pH1, gravity_level)

    y1_pH = replace_zeros(y1_pH)
    y2_pH = replace_zeros(y2_pH)
    y1_gravity = replace_zeros(y1_gravity)
    y2_gravity = replace_zeros(y2_gravity)

    # 计算模糊规则权重
    beta_values = calculate_rule_weights(y1_pH, y2_pH, y1_gravity, y2_gravity)

    # 构建设计矩阵 Y
    Y = build_design_matrix(beta_values, pH1, gravity_level)

    # 计算线性回归模型系数 P
    P = np.linalg.pinv(Y.T @ Y) @ Y.T @ error

    # 生成网格数据
    pph = np.linspace(11.0, 14.0, 10)
    gravity_range = np.linspace(0, 200, 10)
    pH_grid, gravity_grid = generate_grid(pph, gravity_range)

    # 计算网格上的隶属度函数
    pH_low_membership, pH_high_membership, gravity_low_membership, gravity_high_membership = calculate_membership_functions_on_grid(pH_grid, gravity_grid)

    pH_low_membership = replace_zeros(pH_low_membership)
    pH_high_membership = replace_zeros(pH_high_membership)
    gravity_low_membership = replace_zeros(gravity_low_membership)
    gravity_high_membership = replace_zeros(gravity_high_membership)


    # 计算去除效率（解模糊化）
    removal_efficiency = (pH_low_membership * gravity_low_membership * (P[0] + P[4] * pH_grid + P[8] * gravity_grid) +
                          pH_low_membership * gravity_high_membership * (P[1] + P[5] * pH_grid + P[9] * gravity_grid) +
                          pH_high_membership * gravity_low_membership * (P[2] + P[6] * pH_grid + P[10] * gravity_grid) +
                          pH_high_membership * gravity_high_membership * (P[3] + P[7] * pH_grid + P[11] * gravity_grid)) / (
                          pH_low_membership * gravity_low_membership +
                          pH_low_membership * gravity_high_membership +
                          pH_high_membership * gravity_low_membership +
                          pH_high_membership * gravity_high_membership)

    # 插值部分
    tt = np.arange(11.5, 13.5, 0.01)
    y_remo = 1 - (removal_efficiency / 0.0005)  # 对去除效率进行线性变换
    ynew2 = perform_interpolation(pph, y_remo, tt)  # 在新的 pH 值区间上进行插值

    # 绘制插值后的结果
    plot_3d_surface(tt, gravity_range, ynew2.T)

    # # 绘制结果
    # plot_3d_surface(pph, gravity_range, removal_efficiency)

# 执行主函数
if __name__ == "__main__":
    main()
