import matplotlib.pyplot as plt
import numpy as np

data = {
    '初生代谢物': {'脂质': 325, '氨基酸及其衍生物': 303, '有机酸': 122, '核苷酸及其衍生物': 76},
    '次生代谢物': {
        '黄酮': 463, '萜类': 359,
        '酚酸类': 311, '生物碱': 248, '木脂素和香豆素': 147,
        '醌类': 30, '鞣质': 19, '甾体': 14
    },
    '其他类': {'其他类': 416}
}

palette = {
    '初生代谢物': plt.cm.Blues(np.linspace(0.3, 0.8, 4)),
    '次生代谢物': plt.cm.Greens(np.linspace(0.2, 0.8, 8)),
    '其他类': ['#999999']
}

# 创建大尺寸画布
fig, ax = plt.subplots(figsize=(20, 16))  # 进一步增大画布尺寸
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 外层：次分类（小类） ==========
all_sub_values = []
all_sub_colors = []
for main_cat, colors in palette.items():
    all_sub_values.extend(data[main_cat].values())
    all_sub_colors.extend(colors)

# 绘制外层（小类）
wedges_sub, texts_sub = ax.pie(
    all_sub_values,
    radius=1.4,
    colors=all_sub_colors,
    startangle=90,
    wedgeprops=dict(width=0.4, edgecolor='w', alpha=0.9)
)

# ========== 内层：主分类（大类） ==========
main_values = [sum(d.values()) for d in data.values()]
main_labels = [f'{k}\n({v})' for k, v in zip(data.keys(), main_values)]

wedges_main, texts_main = ax.pie(
    main_values,
    radius=1.0,  # 调整内圈半径
    colors=[c[0] for c in palette.values()],
    startangle=90,
    wedgeprops=dict(width=0.3, edgecolor='w', alpha=0.9),
    labels=main_labels,
    labeldistance=0.75,
    textprops=dict(fontsize=14, weight='bold')  # 增大标签字号
)

# ========== 优化图例显示 ==========
legend_elements = []
for main_cat, sub_dict in data.items():
    # 主分类图例（使用全角空格对齐）
    legend_elements.append(plt.Line2D([0], [0],
                                      marker='s', color='w', markersize=18,
                                      markerfacecolor=palette[main_cat][0],
                                      label=f'　{main_cat}'))  # 全角空格

    # 子分类图例（添加缩进）
    for sub_cat, value in sub_dict.items():
        legend_elements.append(plt.Line2D([0], [0],
                                          marker='s', color='w', markersize=12,
                                          markerfacecolor=palette[main_cat][list(sub_dict.keys()).index(sub_cat)],
                                          label=f'　　　{sub_cat} ({value})'))  # 三级缩进

# 创建分栏图例
legend = ax.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),  # 上对齐
    fontsize=12,  # 调小字号
    frameon=False,
    title='代谢物分类图例',
    title_fontsize=14,
    handletextpad=0.5,  # 调整图标与文本间距
    borderpad=0.8,  # 调整边框间距
    labelspacing=0.8  # 调整行间距
)

# 设置图例透明度（避免被背景覆盖）
legend.get_frame().set_alpha(0.5)

# 调整布局参数
plt.title('代谢物分类层级分布\n', fontsize=20, y=0.93, weight='bold')
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.78)  # 为图例留出更多空间

# ========== 保存高清图片 ==========
plt.savefig(
    'metabolites_classification.png',
    dpi=600,  # 印刷级分辨率
    bbox_inches='tight',  # 包含所有元素
    pad_inches=0.3,  # 增加边缘留白
    facecolor='white'  # 设置背景为白色
)

plt.show()