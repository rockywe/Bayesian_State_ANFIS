import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 定义三角隶属函数
def triangular_membership_function(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# 模拟理论模型的值
simulated_values_1 = np.array([0.6010, 0.6923, 0.7799, 0.8534, 0.8876, 0.9078, 0.9212, 0.9309, 0.9382, 0.9439])
simulated_values_2 = np.array([0.5834, 0.6737, 0.7631, 0.8401, 0.8766, 0.8984, 0.9130, 0.9236, 0.9316, 0.9379])
simulated_values_3 = np.array([0.4918, 0.5659, 0.6547, 0.7454, 0.7942, 0.8256, 0.8478, 0.8645, 0.8775, 0.8880])

# 实际去除效率
actual_efficiency_135 = np.array([0.5976, 0.8068, 0.8996, 0.9222, 0.9882, 0.9918, 0.9932, 0.994, 0.993, 0.9922])
actual_efficiency_125 = np.array([0.4642, 0.6564, 0.8052, 0.9198, 0.9074, 0.9116, 0.9148, 0.9106, 0.9078, 0.9056])
actual_efficiency_115 = np.array([0.3332, 0.4426, 0.4976, 0.5312, 0.5666, 0.5922, 0.5788, 0.5832, 0.565, 0.5634])

# T-S模糊模型误差
error_135 = 0.0005 * (1 - actual_efficiency_135)
error_125 = 0.0005 * (1 - actual_efficiency_125)
error_115 = 0.0005 * (1 - actual_efficiency_115)

error_1 = error_135 - 5e-4 * (1 - simulated_values_1)
error_2 = error_125 - 5e-4 * (1 - simulated_values_2)
error_3 = error_115 - 5e-4 * (1 - simulated_values_3)

# 输入参数
ph_values = np.array([13.5] * 10 + [12.5] * 10 + [11.5] * 10)
h2o2_concentrations = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6] * 3)
total_error = np.concatenate([error_1, error_2, error_3])

# 隶属函数计算
ph_low_membership = triangular_membership_function(ph_values, [10.0, 11.5, 12.0])
ph_high_membership = triangular_membership_function(ph_values, [11.95, 13.5, 14.0])

h2o2_low_membership = triangular_membership_function(h2o2_concentrations, [0, 0.2, 0.5])
h2o2_medium_membership = triangular_membership_function(h2o2_concentrations, [0.4, 0.65, 0.85])
h2o2_high_membership = triangular_membership_function(h2o2_concentrations, [0.8, 1.2, 1.70])

# 模糊规则
rule_1 = ph_low_membership * h2o2_low_membership
rule_2 = ph_low_membership * h2o2_medium_membership
rule_3 = ph_low_membership * h2o2_high_membership
rule_4 = ph_high_membership * h2o2_low_membership
rule_5 = ph_high_membership * h2o2_medium_membership
rule_6 = ph_high_membership * h2o2_high_membership

# 计算β值
rule_sum = rule_1 + rule_2 + rule_3 + rule_4 + rule_5 + rule_6
beta_1 = rule_1 / rule_sum
beta_2 = rule_2 / rule_sum
beta_3 = rule_3 / rule_sum
beta_4 = rule_4 / rule_sum
beta_5 = rule_5 / rule_sum
beta_6 = rule_6 / rule_sum

beta_ph_1 = beta_1 * ph_values
beta_ph_2 = beta_2 * ph_values
beta_ph_3 = beta_3 * ph_values
beta_ph_4 = beta_4 * ph_values
beta_ph_5 = beta_5 * ph_values
beta_ph_6 = beta_6 * ph_values

beta_h2o2_1 = beta_1 * h2o2_concentrations
beta_h2o2_2 = beta_2 * h2o2_concentrations
beta_h2o2_3 = beta_3 * h2o2_concentrations
beta_h2o2_4 = beta_4 * h2o2_concentrations
beta_h2o2_5 = beta_5 * h2o2_concentrations
beta_h2o2_6 = beta_6 * h2o2_concentrations

# 构建设计矩阵Y
Y_matrix = np.column_stack([
    beta_1, beta_2, beta_3, beta_4, beta_5, beta_6,
    beta_ph_1, beta_ph_2, beta_ph_3, beta_ph_4, beta_ph_5, beta_ph_6,
    beta_h2o2_1, beta_h2o2_2, beta_h2o2_3, beta_h2o2_4, beta_h2o2_5, beta_h2o2_6
])

# 计算P矩阵
P_matrix = np.linalg.pinv(Y_matrix.T @ Y_matrix) @ Y_matrix.T @ total_error

# 生成网格
ph_grid = np.linspace(11.5, 13.5, 10)
h2o2_grid = np.linspace(0.05, 1.6, 10)
ph_mesh, h2o2_mesh = np.meshgrid(ph_grid, h2o2_grid)

ph_low_mesh = triangular_membership_function(ph_mesh, [10.0, 11.5, 11.8])
ph_high_mesh = triangular_membership_function(ph_mesh, [11.75, 13.5, 14.0])

h2o2_low_mesh = triangular_membership_function(h2o2_mesh, [0, 0.2, 0.5])
h2o2_medium_mesh = triangular_membership_function(h2o2_mesh, [0.4, 0.65, 0.85])
h2o2_high_mesh = triangular_membership_function(h2o2_mesh, [0.8, 1.2, 1.70])

rule_mesh_1 = ph_low_mesh * h2o2_low_mesh
rule_mesh_2 = ph_low_mesh * h2o2_medium_mesh
rule_mesh_3 = ph_low_mesh * h2o2_high_mesh
rule_mesh_4 = ph_high_mesh * h2o2_low_mesh
rule_mesh_5 = ph_high_mesh * h2o2_medium_mesh
rule_mesh_6 = ph_high_mesh * h2o2_high_mesh

# 模糊输出
fuzzy_output = (rule_mesh_1 * (P_matrix[0] + P_matrix[6] * ph_mesh + P_matrix[12] * h2o2_mesh) +
                rule_mesh_2 * (P_matrix[1] + P_matrix[7] * ph_mesh + P_matrix[13] * h2o2_mesh) +
                rule_mesh_3 * (P_matrix[2] + P_matrix[8] * ph_mesh + P_matrix[14] * h2o2_mesh) +
                rule_mesh_4 * (P_matrix[3] + P_matrix[9] * ph_mesh + P_matrix[15] * h2o2_mesh) +
                rule_mesh_5 * (P_matrix[4] + P_matrix[10] * ph_mesh + P_matrix[16] * h2o2_mesh) +
                rule_mesh_6 * (P_matrix[5] + P_matrix[11] * ph_mesh + P_matrix[17] * h2o2_mesh)) / (
                rule_mesh_1 + rule_mesh_2 + rule_mesh_3 + rule_mesh_4 + rule_mesh_5 + rule_mesh_6)

# 插值计算
ph_interpolation = np.arange(11.5, 13.5, 0.01)
removal_efficiency = 1 - (fuzzy_output / 0.0005)
interpolated_efficiency = np.zeros((len(ph_interpolation), removal_efficiency.shape[0]))

for i in range(removal_efficiency.shape[0]):
    cubic_spline = CubicSpline(ph_grid, removal_efficiency[i, :])
    interpolated_efficiency[:, i] = cubic_spline(ph_interpolation)

# 绘制3D表面图
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
ph_mesh_interpolated, h2o2_mesh_interpolated = np.meshgrid(ph_interpolation, h2o2_grid)
ax1.plot_surface(ph_mesh_interpolated, h2o2_mesh_interpolated, interpolated_efficiency.T * 100, cmap='viridis')
ax1.set_xlabel('Absorbent pH', fontsize=23, fontname='Times New Roman')
ax1.set_ylabel('${H_2O_2}$ (mol/L)', fontsize=23, fontname='Times New Roman')
ax1.set_zlabel('NO removal efficiency (%)', fontsize=23, fontname='Times New Roman')
ax1.set_title('(a)', fontsize=35, fontname='Times New Roman')
ax1.view_init(elev=30, azim=120)

plt.tight_layout()
plt.show()
