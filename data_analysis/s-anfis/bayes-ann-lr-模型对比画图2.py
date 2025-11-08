import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import json
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kaleido

mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica World', 'Arial', 'Arial Unicode MS']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['path.simplify'] = True
plt.rcParams['path.snap'] = True

# --- 1. 加载您的原始数据 ---
file_path = './模型结果.xlsx'
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
axes[0].set_ylabel(r'$\mathrm{R}^2$', fontsize=20)
axes[0].set_xlabel('')
axes[0].set_ylim(0.8, 1.05)
axes[0].tick_params(axis='y', which='major', direction='out', length=6, color='black', labelsize=16, left=True, right=False)
axes[0].tick_params(axis='x', length=0)  # 隐藏刻度线
axes[0].xaxis.set_tick_params(pad=12, labelsize=16)     # 调整标签和轴的间距，6 相当于 length=6 时的距离
axes[0].grid(False)
axes[0].legend(loc='lower left',fontsize=18)
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
axes[1].set_ylabel('RMSE ', fontsize=20)
axes[1].set_xlabel('')
axes[1].tick_params(axis='y', which='major', direction='out', length=6, color='black', labelsize=16, left=True, right=False)
axes[1].tick_params(axis='x', length=0)  # 隐藏刻度线
axes[1].xaxis.set_tick_params(pad=12, labelsize=16)     # 调整标签和轴的间距，6 相当于 length=6 时的距离
axes[1].grid(False)
axes[1].get_legend().remove()
for i in range(len(model_order) - 1):
    axes[1].axvline(i + 0.5, color='gray', linestyle='--', linewidth=1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('model_comparison.pdf', bbox_inches='tight')
plt.show()



# 你的原始配置
model_order = ['LR', 'ANN', 'ANFIS', 'BS-ANFIS']

# 为每个模型设置颜色
model_colors = {
    'LR': '#83B4A3',
    'ANN': '#DC9A7A', 
    'ANFIS': '#99A3C1',
    'BS-ANFIS': '#D3573E'
}

# Train和Test的透明度设置
alpha_train = 0.9  # Train较深
alpha_test = 0.5   # Test较浅

# 创建包含两个子图的图表（1行2列）
fig = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.1,  # 子图之间的水平间距
    vertical_spacing=0.1,
    specs=[[{"type": "box"}, {"type": "box"}]]  # 指定都是箱线图
)

# ========== 子图(a): R² Score ==========
for i, model in enumerate(model_order):
    # Train数据
    train_data = df_r2_long[(df_r2_long['Model'] == model) & (df_r2_long['DataSet'] == 'Train')]['R²'].values
    test_data = df_r2_long[(df_r2_long['Model'] == model) & (df_r2_long['DataSet'] == 'Test (Validation)')]['R²'].values
    
    # 计算x位置
    x_pos_train = i * 2
    x_pos_test = i * 2 + 0.8
    
    # 添加Train箱线图到子图1
    fig.add_trace(go.Box(
        y=train_data,
        x=[x_pos_train] * len(train_data),
        name=f"{model} Train",
        boxpoints="all",
        width=0.3,
        whiskerwidth=0.4,
        jitter=0.2,
        pointpos=-1.5,
        line=dict(color=model_colors[model], width=2),
        fillcolor=f"rgba{tuple(int(model_colors[model][i:i+2], 16) for i in (1, 3, 5)) + (alpha_train,)}",
        showlegend=False,
        marker=dict(
            color=f"rgba{tuple(int(model_colors[model][i:i+2], 16) for i in (1, 3, 5)) + (alpha_train,)}",
            size=5,
            line=dict(color=model_colors[model], width=1)
        ),
        boxmean="sd",
        notched=True,
    ), row=1, col=1)  # 指定子图位置
    
    # 添加Test箱线图到子图1
    fig.add_trace(go.Box(
        y=test_data,
        x=[x_pos_test] * len(test_data),
        name=f"{model} Test",
        boxpoints="all",
        width=0.3,
        whiskerwidth=0.4,
        jitter=0.2,
        pointpos=-1.5,
        line=dict(color=model_colors[model], width=2),
        fillcolor=f"rgba{tuple(int(model_colors[model][i:i+2], 16) for i in (1, 3, 5)) + (alpha_test,)}",
        showlegend=False,
        marker=dict(
            color=f"rgba{tuple(int(model_colors[model][i:i+2], 16) for i in (1, 3, 5)) + (alpha_test,)}",
            size=5,
            line=dict(color=model_colors[model], width=1)
        ),
        boxmean="sd",
        notched=True,
    ), row=1, col=1)

# ========== 子图(b): RMSE ==========
# 先获取RMSE数据的范围
all_rmse_values = []
for i, model in enumerate(model_order):
    # RMSE数据
    train_data = df_rmse_long[(df_rmse_long['Model'] == model) & (df_rmse_long['DataSet'] == 'Train')]['RMSE (ppm)'].values
    test_data = df_rmse_long[(df_rmse_long['Model'] == model) & (df_rmse_long['DataSet'] == 'Test (Validation)')]['RMSE (ppm)'].values
    all_rmse_values.extend(train_data)
    all_rmse_values.extend(test_data)
    
    # 计算x位置
    x_pos_train = i * 2
    x_pos_test = i * 2 + 0.8
    
    # 添加Train箱线图到子图2
    fig.add_trace(go.Box(
        y=train_data,
        x=[x_pos_train] * len(train_data),
        name=f"{model} Train RMSE",
        boxpoints="all",
        width=0.3,
        whiskerwidth=0.4,
        jitter=0.2,
        pointpos=-1.5,
        line=dict(color=model_colors[model], width=2),
        fillcolor=f"rgba{tuple(int(model_colors[model][i:i+2], 16) for i in (1, 3, 5)) + (alpha_train,)}",
        showlegend=False,
        marker=dict(
            color=f"rgba{tuple(int(model_colors[model][i:i+2], 16) for i in (1, 3, 5)) + (alpha_train,)}",
            size=5,
            line=dict(color=model_colors[model], width=1)
        ),
        boxmean="sd",
        notched=True,
    ), row=1, col=2)  # 指定子图位置
    
    # 添加Test箱线图到子图2
    fig.add_trace(go.Box(
        y=test_data,
        x=[x_pos_test] * len(test_data),
        name=f"{model} Test RMSE",
        boxpoints="all",
        width=0.3,
        whiskerwidth=0.4,
        jitter=0.2,
        pointpos=-1.5,
        line=dict(color=model_colors[model], width=2),
        fillcolor=f"rgba{tuple(int(model_colors[model][i:i+2], 16) for i in (1, 3, 5)) + (alpha_test,)}",
        showlegend=False,
        marker=dict(
            color=f"rgba{tuple(int(model_colors[model][i:i+2], 16) for i in (1, 3, 5)) + (alpha_test,)}",
            size=5,
            line=dict(color=model_colors[model], width=1)
        ),
        boxmean="sd",
        notched=True,
    ), row=1, col=2)

# 计算RMSE的y轴范围
rmse_min = min(all_rmse_values) * 0.9
rmse_max = max(all_rmse_values) * 1.1

# 更新布局
fig.update_layout(
    # 整体图表设置
    height=500,  # 图表高度
    width=1400,  # 图表宽度（两个子图的总宽度）
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
    font=dict(color='black'),
    margin=dict(t=60, b=80, l=80, r=50),
    
    # 子图标题样式
    title_font=dict(size=16, color='black')
)

# 更新x轴（两个子图）
for i in range(1, 3):
    fig.update_xaxes(
        tickfont=dict(size=16, color='black'),
        tickangle=0,
        tickmode='array',
        tickvals=[j*2 + 0.4 for j in range(len(model_order))],
        ticktext=model_order,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        mirror=False,
        zeroline=False,
        tickcolor='black',
        ticks='',
        row=1, col=i
    )

# 更新y轴 - 子图1（R²）
fig.update_yaxes(
    title=dict(
        text="R²",
        font=dict(size=20, color='black'),
        standoff=20
    ),
    tickfont=dict(size=16, color='black'),
    range=[0.8, 1.05],
    showgrid=False,
    ticklen=6,
    ticks="outside",
    tickcolor='black',
    linecolor='black',
    linewidth=1.5,
    showline=True,
    mirror=False,
    zeroline=False,
    row=1, col=1
)

# 更新y轴 - 子图2（RMSE）
fig.update_yaxes(
    title=dict(
        text="RMSE (ppm)",
        font=dict(size=20, color='black'),
        standoff=20
    ),
    tickfont=dict(size=16, color='black'),
    range=[rmse_min, rmse_max],  # 使用计算出的范围
    showgrid=False,
    ticklen=6,
    ticks="outside",
    tickcolor='black',
    linecolor='black',
    linewidth=1.5,
    showline=True,
    mirror=False,
    zeroline=False,
    row=1, col=2
)

# 在更新完y轴后，添加Train/Test标注
# 子图1（R²）的标注 - 使用固定位置1.03
r2_label_y = 1.03  # 固定在1.03位置
rmse_label_y = rmse_max * 0.9413  # RMSE使用最大值的98%

text_annotations = []
for i, model in enumerate(model_order):
    x_pos_train = i * 2
    x_pos_test = i * 2 + 0.8
    
    # R²图的标注
    text_annotations.extend([
        dict(
            x=x_pos_train,
            y=r2_label_y,
            text="Train",
            showarrow=False,
            font=dict(size=14, color='black'),
            xanchor='center',
            yanchor='bottom',
            xref='x',
            yref='y'
        ),
        dict(
            x=x_pos_test,
            y=r2_label_y,
            text="Test",
            showarrow=False,
            font=dict(size=14, color='black'),
            xanchor='center',
            yanchor='bottom',
            xref='x',
            yref='y'
        )
    ])
    
    # RMSE图的标注
    text_annotations.extend([
        dict(
            x=x_pos_train,
            y=rmse_label_y,
            text="Train",
            showarrow=False,
            font=dict(size=14, color='black'),
            xanchor='center',
            yanchor='bottom',
            xref='x2',
            yref='y2'
        ),
        dict(
            x=x_pos_test,
            y=rmse_label_y,
            text="Test",
            showarrow=False,
            font=dict(size=14, color='black'),
            xanchor='center',
            yanchor='bottom',
            xref='x2',
            yref='y2'
        )
    ])

# 将标注添加到布局中
fig.update_layout(annotations=text_annotations)

# 为每个子图添加边框
for i in range(1, 3):
    # 上边框
    fig.add_shape(
        type="line",
        xref=f"x{i if i > 1 else ''} domain",
        yref=f"y{i if i > 1 else ''} domain",
        x0=0, y0=1, x1=1, y1=1,
        line=dict(color="black", width=1.5)
    )
    # 右边框
    fig.add_shape(
        type="line",
        xref=f"x{i if i > 1 else ''} domain",
        yref=f"y{i if i > 1 else ''} domain",
        x0=1, y0=0, x1=1, y1=1,
        line=dict(color="black", width=1.5)
    )

# 添加模型之间的分隔线
for i in range(len(model_order) - 1):
    # 子图1的分隔线
    fig.add_vline(
        x=i*2 + 1.4,
        line_color="gray",
        line_dash="dash",
        line_width=1,
        row=1, col=1
    )
    # 子图2的分隔线
    fig.add_vline(
        x=i*2 + 1.4,
        line_color="gray",
        line_dash="dash",
        line_width=1,
        row=1, col=2
    )

# 显示图表
fig.show()

# 保存图表
try:
    fig.write_image("boxplot_combined.png", width=1400, height=500, scale=8.33)
    print("✓ PNG保存成功: boxplot_combined.png")
    
    fig.write_image("boxplot_combined.pdf", width=1400, height=500)
    print("✓ PDF保存成功: boxplot_combined.pdf")
except Exception as e:
    print(f"保存失败: {e}")
    fig.write_html("boxplot_combined.html")
    print("✓ HTML保存成功: boxplot_combined.html")