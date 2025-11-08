import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 加载您的原始数据 ---
file_path = './data/模型结果.xlsx'
try:
    df_r2 = pd.read_excel(file_path, sheet_name='Sheet1')
    df_rmse = pd.read_excel(file_path, sheet_name='Sheet2')
    print("成功加载原始Excel数据。")
except FileNotFoundError:
    print("错误：找不到 '各个模型对比数据.xlsx' 文件。请确保文件在正确的路径下。")
    df_r2 = pd.DataFrame()
    df_rmse = pd.DataFrame()


# --- 2. 数据重塑 (Melt/Tidy) ---
# 使用修正后的拆分逻辑
def reshape_data(df, value_name):
    # 将连字符'-'替换为下划线'_'，方便后续统一处理
    df.columns = [col.replace('-', '_') for col in df.columns]

    # 转换（Melt）
    df_long = df.melt(var_name='Model_DataSet', value_name=value_name)

    # --- 这是修正的核心部分 ---
    # 使用 rsplit 从右边只分割一次，确保模型名称的完整性
    split_cols = df_long['Model_DataSet'].str.rsplit('_', n=1, expand=True)
    df_long['Model'] = split_cols[0]
    df_long['DataSet'] = split_cols[1]

    # 格式化名称
    df_long['Model'] = df_long['Model'].str.replace('_', '-').str.upper()  # 将 bs_anfis 变回 BS-ANFIS
    df_long['DataSet'] = df_long['DataSet'].str.capitalize()
    df_long['DataSet'] = df_long['DataSet'].replace({'Test': 'Test (Validation)'})

    return df_long


df_r2_long = reshape_data(df_r2.copy(), 'R²')
df_rmse_long = reshape_data(df_rmse.copy(), 'RMSE (ppm)')

# --- 3. 绘图 ---
# (这部分代码无需修改)
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=100)
# fig.suptitle('Model Performance Comparison (Based on Original Data)', fontsize=20, y=1.02)

model_order = ['LR', 'ANN', 'ANFIS', 'BS-ANFIS']
palette = {'Train': '#084c54', 'Test (Validation)': '#f5f0e1'}

# 子图 (a): R² Score
sns.boxplot(
    ax=axes[0], data=df_r2_long, x='Model', y='R²', hue='DataSet', order=model_order,
    palette=palette, showmeans=True,
    meanprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': '6'},
    boxprops={'edgecolor': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1.5},
    capprops={'color': 'black', 'linewidth': 1.5},
    medianprops={'color': 'black', 'linewidth': 1.5},
    flierprops={'marker': '+', 'markeredgecolor': 'gray'}
)
axes[0].set_title('(a) R² Score', fontsize=16)
axes[0].set_xlabel('Model', fontsize=14)
axes[0].set_ylabel('R²', fontsize=14)
axes[0].set_ylim(0.8, 1.05)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].legend(title='Data Set', loc='lower left')
for i in range(len(model_order) - 1):
    axes[0].axvline(i + 0.5, color='gray', linestyle='--', linewidth=1)

# 子图 (b): Root Mean Squared Error (RMSE)
sns.boxplot(
    ax=axes[1], data=df_rmse_long, x='Model', y='RMSE (ppm)', hue='DataSet', order=model_order,
    palette=palette, showmeans=True,
    meanprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': '6'},
    boxprops={'edgecolor': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1.5},
    capprops={'color': 'black', 'linewidth': 1.5},
    medianprops={'color': 'black', 'linewidth': 1.5},
    flierprops={'marker': '+', 'markeredgecolor': 'gray'}
)
axes[1].set_title('(b) Root Mean Squared Error (RMSE)', fontsize=16)
axes[1].set_xlabel('Model', fontsize=14)
axes[1].set_ylabel('RMSE ', fontsize=14)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].get_legend().remove()
for i in range(len(model_order) - 1):
    axes[1].axvline(i + 0.5, color='gray', linestyle='--', linewidth=1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
