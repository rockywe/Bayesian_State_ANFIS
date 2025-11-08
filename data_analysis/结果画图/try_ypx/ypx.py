import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.multicomp as multi

# 设置中文字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取Excel文件并选择Sheet2工作表
# file_path = '虫试11.1.xlsx'
# file_path = '11.24虫试.xlsx'
file_path = '11.24虫试1.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# # 选择需要绘制的数据列
# concentrations = ["浓度30μM", "浓度60μM 12H", "浓度90μM 12H", "CK 12H"]
# initial_weights_cols = ["浓度30μM_初始体重", "浓度60μM 12H_初始体重", "浓度90μM 12H_初始体重", "CK 12H_初始体重"]
# final_weights_cols = ["浓度30μM_饲喂4天体重", "浓度60μM 12H_饲喂4天体重", "浓度90μM 12H_饲喂4天体重", "CK 12H_饲喂4天体重"]
#
# # 创建4个图表，分别比较每个浓度的初始体重与4天后体重的变化
# for i, conc in enumerate(concentrations):
#     plt.figure(figsize=(10, 6))
#     plt.plot(df.index, df[initial_weights_cols[i]], marker='o', linestyle='-', label=f'{conc} 初始体重', color='blue')
#     plt.plot(df.index, df[final_weights_cols[i]], marker='s', linestyle='--', label=f'{conc} 饲喂4天体重', color='red')
#
#     # 设置图表标题和标签
#     plt.title(f'{conc} 初始体重与饲喂4天体重的对比')
#     plt.xlabel('样本编号')
#     plt.ylabel('体重 (g)')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()



# 选择需要绘制的数据列
# concentrations = ["浓度30μM", "浓度60μM 12H", "浓度90μM 12H", "CK 12H"]
# concentrations = ["CK", "100", "200"]
concentrations = ["CK", "100"]
# final_weights_cols = ["浓度30μM_增长", "浓度60μM_增长", "浓度90μM_增长", "CK_增长"]
# final_weights_cols = ["CK1H增量", "1001H增量",""2001H增量""]
final_weights_cols = ["CK3H增量", "1003H增量"]

# 重组数据以便于绘制箱线图
boxplot_data = pd.DataFrame()
for i, conc in enumerate(concentrations):
    temp_df = pd.DataFrame({
        '浓度': conc,
        '体重 (g)': df[final_weights_cols[i]]
    })
    boxplot_data = pd.concat([boxplot_data, temp_df], axis=0, ignore_index=True)

# 绘制箱线图并标注数据点
plt.figure(figsize=(8, 5))
sns.boxplot(x='浓度', y='体重 (g)', data=boxplot_data, palette='Set3', showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
sns.swarmplot(x='浓度', y='体重 (g)', data=boxplot_data, color='black', size=4, alpha=0.8)

# 设置图表标题和标签
# plt.title('不同浓度下饲喂4天后体重的对比', fontsize=14, pad=15)
plt.title('不同浓度下饲喂体重的对比', fontsize=14, pad=15)
plt.xlabel('浓度', fontsize=12, labelpad=10)
plt.ylabel('幼虫体重 (g)', fontsize=12, labelpad=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout(pad=1.0)
plt.show()