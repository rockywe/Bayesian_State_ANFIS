import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica World', 'Arial', 'Arial Unicode MS', 'SimHei']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['path.simplify'] = True
plt.rcParams['path.snap'] = True
# 正确的数据
rounds = list(range(1, 102))

# R²值
r2_values = [
    -3.278513, 0.627957, -3.278513, -3.278513, 0.845915, 0.420166, -3.10784,
    -2.694056, 0.749229, -3.266108, 0.923461, 0.801276, -3.278513, 0.657215,
    0.913802, 0.302653, -3.278513, 0.945951, 0.829754, 0.903885, 0.881014,
    -2.947214, 0.405902, -3.278513, 0.337119, 0.951241, 0.745852, 0.847563,
    0.839667, 0.578331, 0.866842, 0.727123, 0.723919, -0.45773, 0.797917,
    -3.278513, 0.790655, 0.356944, -0.277006, 0.824124, 0.030105, 0.851987,
    0.511543, -1.360071, 0.67318, 0.620906, 0.880231, 0.931462, 0.925642,
    0.786574, 0.858468, -2.901539, 0.888659, 0.784124, 0.940242, 0.185714,
    0.947101, -3.278513, 0.270711, 0.835884, 0.8795, -2.545332, -2.278513,
    -2.428064, 0.088508, 0.513313, 0.299917, 0.960099, -2.491174, 0.824627,
    -3.242094, 0.593132, -0.058648, 0.237979, -10000000000, -2.753818,
    0.897868, 0.776806, -3.278513, 0.964626, 0.955671, 0.696213, 0.915873,
    0.636656, 0.947218, 0.339654, 0.818081, 0.902197, -3.627585, -3.278513,
    0.849981, -2.547325, 0.919382, 0.910112, -2.481968, -0.029671, 0.693911,
    0.429397, 0.848014, -2.029359, 0.701511
]

# RMSE值
rmse_values = [
    2.231434, 0.703236, 2.231434, 2.231434, 0.530634, 0.89618, 2.198063,
    2.073763, 0.637722, 2.228906, 0.350674, 0.572252, 2.231434, 0.712254,
    0.367118, 1.021, 2.231434, 0.283284, 0.526, 0.378, 0.427, 2.137, 0.895,
    2.231434, 0.982, 0.266, 0.643, 0.536, 0.549, 0.793, 0.472, 0.623, 0.629,
    1.292, 0.58, 2.231434, 0.59, 0.958, 1.168, 0.544, 1.255, 0.522, 0.827,
    1.728, 0.676, 0.719, 0.436, 0.324, 0.336, 0.605, 0.497, 2.123, 0.455,
    0.606,  0.606, 1.077, 0.287, 2.231434, 1.038, 0.525, 0.439, 2.045, 2.231434,
    2.02, 1.24, 0.825, 1.021, 0.234, 2.032, 0.543, 2.219, 0.774, 1.267, 1.05,
    1000000, 2.095, 0.397, 0.592, 2.231434, 0.204, 0.23, 0.658, 0.365, 0.726,
    0.286, 0.98, 0.539, 0.383, 2.311, 2.231434, 0.515, 2.046, 0.357, 0.373,
    2.034, 1.258, 0.665, 0.887, 0.516, 1.911, 0.675
]

# 确保数据长度一致
min_length = min(len(rounds), len(r2_values), len(rmse_values))
rounds = rounds[:min_length]
r2_values = r2_values[:min_length]
rmse_values = rmse_values[:min_length]

target_x = 100

# 筛选最优r2点（单调递增）
optimal_rounds = []
optimal_r2 = []
optimal_rmse = []
current_max = -float('inf')

for i, (round_num, r2, rmse) in enumerate(zip(rounds, r2_values, rmse_values)):
    if r2 > current_max:
        optimal_rounds.append(round_num)
        optimal_r2.append(r2)
        optimal_rmse.append(rmse)
        current_max = r2


ext_optimal_rounds = optimal_rounds.copy()
ext_optimal_r2 = optimal_r2.copy()
ext_optimal_rmse = optimal_rmse.copy()
if ext_optimal_rounds[-1] < target_x:
    ext_optimal_rounds.append(target_x)
    ext_optimal_r2.append(ext_optimal_r2[-1])
    ext_optimal_rmse.append(ext_optimal_rmse[-1])

# 创建1行2列的图形布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

r2_scatter_c ='#9ecae1'    # 浅蓝
r2_line_c = '#1f78b4'   # 深蓝
rmse_scatter_c = '#fdd0a2' # 浅橙
rmse_line_c = '#e6550d'  # 深橙

# 第一张图：R²值变化及最优进步轨迹
ax1.scatter(rounds, r2_values, color=r2_scatter_c, s=24, alpha=0.8,
edgecolor='white', linewidths=0.5, label='Iteration value')
ax1.plot(ext_optimal_rounds, ext_optimal_r2, color=r2_line_c, marker='o',markersize=6, linewidth=3, label='Best value')
ax1.set_xlabel('Iteration Step', fontsize=22)
ax1.set_ylabel('R2 during iterations', fontsize=22)
ax1.set_xlim(0, 100)
ax1.set_ylim(0.7, 1.0)
ax1.legend(fontsize=14, loc='upper right')

# 第二张图：RMSE值变化及对应最优点的RMSE
ax2.scatter(rounds, rmse_values, color=rmse_scatter_c, s=24, alpha=0.8,
edgecolor='white', linewidths=0.5, label='Iteration value')
ax2.plot(ext_optimal_rounds, ext_optimal_rmse, color=rmse_line_c, marker='s',
markersize=6, linewidth=3, label='Best value')
ax2.set_xlabel('Iteration Step', fontsize=22)
ax2.set_ylabel('RMSE during iterations', fontsize=22)
ax2.set_xlim(0, 100)
ax2.set_ylim(0.2, 1.0)
ax2.legend(fontsize=14, loc='upper right')

plt.tight_layout(pad=3.0)
plt.show()

# 输出详细统计信息
print("=" * 60)
print("最优进步点详细统计:")
print("=" * 60)
print("轮次\tR²值\t\tRMSE值\t\t改进情况")
print("-" * 60)

for i in range(len(optimal_rounds)):
    rnd = optimal_rounds[i]
    r2 = optimal_r2[i]
    rmse = optimal_rmse[i]

    if i == 0:
        improvement = "初始点"
    else:
        prev_r2 = optimal_r2[i - 1]
        prev_rmse = optimal_rmse[i - 1]
        r2_improve = r2 - prev_r2
        rmse_improve = prev_rmse - rmse  # RMSE降低是好的
        improvement = f"R²+{r2_improve:.4f}, RMSE-{rmse_improve:.4f}"

    print(f"{rnd}\t{r2:.6f}\t{rmse:.6f}\t{improvement}")



print("=" * 60)
print(f"总结: 经过{len(optimal_rounds)}次关键改进，R²从{optimal_r2[0]:.4f}提升到{optimal_r2[-1]:.4f}")
print(f"      RMSE从{optimal_rmse[0]:.4f}降低到{optimal_rmse[-1]:.4f}")