import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

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
def calculate_membership_functions(pH_values, h2o2_values):
    y1_pH = triangular_membership_function(pH_values, [10.0, 11.5, 12.0])
    y2_pH = triangular_membership_function(pH_values, [11.95, 13.5, 14.0])
    y1_h2o2 = triangular_membership_function(h2o2_values, [0, 0.2, 0.5])
    y2_h2o2 = triangular_membership_function(h2o2_values, [0.4, 0.65, 0.85])
    y3_h2o2 = triangular_membership_function(h2o2_values, [0.8, 1.2, 1.70])
    return y1_pH, y2_pH, y1_h2o2, y2_h2o2, y3_h2o2

# 计算模糊规则权重
def calculate_rule_weights(y1_pH, y2_pH, y1_h2o2, y2_h2o2, y3_h2o2):
    rule_1 = y1_pH * y1_h2o2
    rule_2 = y1_pH * y2_h2o2
    rule_3 = y1_pH * y3_h2o2
    rule_4 = y2_pH * y1_h2o2
    rule_5 = y2_pH * y2_h2o2
    rule_6 = y2_pH * y3_h2o2

    rule_sum = rule_1 + rule_2 + rule_3 + rule_4 + rule_5 + rule_6

    beta_1 = rule_1 / rule_sum
    beta_2 = rule_2 / rule_sum
    beta_3 = rule_3 / rule_sum
    beta_4 = rule_4 / rule_sum
    beta_5 = rule_5 / rule_sum
    beta_6 = rule_6 / rule_sum

    return beta_1, beta_2, beta_3, beta_4, beta_5, beta_6

# 构建设计矩阵 Y
def build_design_matrix(beta_values, pH_values, h2o2_values):
    beta_pH = [beta * pH_values for beta in beta_values]
    beta_h2o2 = [beta * h2o2_values for beta in beta_values]
    return np.column_stack(list(beta_values) + beta_pH + beta_h2o2)

# 计算插值
def perform_interpolation(pph, y_remo, pH_range):
    interpolated_results = np.zeros((len(pH_range), y_remo.shape[0]))
    for i in range(y_remo.shape[0]):
        cs = CubicSpline(pph, y_remo[i, :])
        interpolated_results[:, i] = cs(pH_range)
    return interpolated_results

# 生成网格数据
def generate_grid(pH_range, h2o2_range):
    pH_grid, h2o2_grid = np.meshgrid(pH_range, h2o2_range)
    return pH_grid, h2o2_grid

# 计算隶属度函数
def calculate_membership_functions_on_grid(pH_grid, h2o2_grid):
    pH_low_membership = triangular_membership_function(pH_grid, [10.0, 11.5, 11.8])
    pH_high_membership = triangular_membership_function(pH_grid, [11.75, 13.5, 14.0])

    h2o2_low_membership = triangular_membership_function(h2o2_grid, [0, 0.2, 0.5])
    h2o2_medium_membership = triangular_membership_function(h2o2_grid, [0.4, 0.65, 0.85])
    h2o2_high_membership = triangular_membership_function(h2o2_grid, [0.8, 1.2, 1.70])

    return pH_low_membership, pH_high_membership, h2o2_low_membership, h2o2_medium_membership, h2o2_high_membership

# 绘制3D表面图
def plot_3d_surface(tt, h22o2, ynew2):
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    tt_mesh, yyy_mesh = np.meshgrid(tt, h22o2)
    ax1.plot_surface(tt_mesh, yyy_mesh, ynew2.T * 100, cmap='viridis')
    ax1.set_xlabel('Absorbent pH', fontsize=23, fontname='Times New Roman')
    ax1.set_ylabel('${H_2O_2}$ (mol/L)', fontsize=23, fontname='Times New Roman')
    ax1.set_zlabel('NO removal efficiency (%)', fontsize=23, fontname='Times New Roman')
    ax1.set_title('(a)', fontsize=35, fontname='Times New Roman')
    ax1.view_init(elev=30, azim=120)
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    yno_1, yno_2, yno_3 = simulate_theoretical_values()
    y_135, y_125, y_115 = actual_removal_efficiency()

    er1 = calculate_error(yno_1, y_135)
    er2 = calculate_error(yno_2, y_125)
    er3 = calculate_error(yno_3, y_115)

    pH1 = np.array([13.5] * 10 + [12.5] * 10 + [11.5] * 10)
    h2o21 = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6] * 3)
    error = np.concatenate([er1, er2, er3])

    y1_pH, y2_pH, y1_h2o2, y2_h2o2, y3_h2o2 = calculate_membership_functions(pH1, h2o21)
    beta_values = calculate_rule_weights(y1_pH, y2_pH, y1_h2o2, y2_h2o2, y3_h2o2)
    Y = build_design_matrix(beta_values, pH1, h2o21)

    P = np.linalg.pinv(Y.T @ Y) @ Y.T @ error

    pph = np.linspace(11.5, 13.5, 10)
    h22o2 = np.linspace(0.05, 1.6, 10)
    z, yyy = generate_grid(pph, h22o2)

    y11 = triangular_membership_function(z, [10.0, 11.5, 11.8])
    y22 = triangular_membership_function(z, [11.75, 13.5, 14.0])

    y33 = triangular_membership_function(yyy, [0, 0.2, 0.5])
    y44 = triangular_membership_function(yyy, [0.4, 0.65, 0.85])
    y55 = triangular_membership_function(yyy, [0.8, 1.2, 1.70])

    yy = (y11 * y33 * (P[0] + P[6] * z + P[12] * yyy) +
          y11 * y44 * (P[1] + P[7] * z + P[13] * yyy) +
          y11 * y55 * (P[2] + P[8] * z + P[14] * yyy) +
          y22 * y33 * (P[3] + P[9] * z + P[15] * yyy) +
          y22 * y44 * (P[4] + P[10] * z + P[16] * yyy) +
          y22 * y55 * (P[5] + P[11] * z + P[17] * yyy)) / (y11 * y33 + y11 * y44 + y11 * y55 + y22 * y33 + y22 * y44 + y22 * y55)

    tt = np.arange(11.5, 13.5, 0.01)
    y_remo = 1 - (yy / 0.0005)
    ynew2 = perform_interpolation(pph, y_remo, tt)

    plot_3d_surface(tt, h22o2, ynew2)

# 执行主函数
if __name__ == "__main__":
    main()
