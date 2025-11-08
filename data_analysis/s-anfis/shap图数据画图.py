import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. 加载您提供的 SHAP 值数据
file_path = './data/shap图数据.xlsx'
shap_df = pd.read_excel(file_path)

# 2. 定义您要求的自定义特征顺序
# 首先，获取基于重要性的默认顺序
default_order = shap_df.abs().mean().sort_values(ascending=False).index.tolist()
# 然后，根据您的要求调整顺序
if 'Rotation_Speed' in default_order:
    default_order.remove('Rotation_Speed')
# 将 'Rotation_Speed' 插入到第二个位置 (索引为 1)
default_order.insert(1, 'Rotation_Speed')
custom_order = default_order

# 使用自定义顺序重新排列 SHAP 值 DataFrame
shap_df_ordered = shap_df[custom_order]


# 3. 创建模拟的特征值数据以重现颜色 (根据更新后的相关性)
mock_features_df = pd.DataFrame(index=shap_df_ordered.index)

# 根据您指定的负相关性来设置 Rotation_Speed
# 高转速 (红色) -> 负 SHAP 值 (点在左侧)
mock_features_df['Rotation_Speed'] = -shap_df_ordered['Rotation_Speed']

# 其他特征的相关性根据其 SHAP 值的主要分布来推断
# 正相关: 高值 (红) -> 正 SHAP (右)
mock_features_df['Temperature_liquid'] = shap_df_ordered['Temperature_liquid']
mock_features_df['Pressure_liquid'] = shap_df_ordered['Pressure_liquid']

# 负相关: 高值 (红) -> 负 SHAP (左)
mock_features_df['Flow_liquid'] = -shap_df_ordered['Flow_liquid']
mock_features_df['Flow_gas'] = -shap_df_ordered['Flow_gas']
mock_features_df['Temperature_gas'] = -shap_df_ordered['Temperature_gas']
mock_features_df['CH2S'] = -shap_df_ordered['CH2S']
mock_features_df['Pressure_gas'] = -shap_df_ordered['Pressure_gas']

# 确保模拟特征的 DataFrame 也遵循相同的自定义顺序
mock_features_ordered = mock_features_df[custom_order]


# 4. 定义 LaTeX 格式的特征标签
latex_label_map = {
    'Flow_liquid': r'$Flow_{\mathrm{liquid}}$',
    'Pressure_liquid': r'$Pressure_{\mathrm{liquid}}$',
    'Temperature_liquid': r'$Temperature_{\mathrm{liquid}}$',
    'Flow_gas': r'$Flow_{\mathrm{gas}}$',
    'Pressure_gas': r'$Pressure_{\mathrm{gas}}$',
    'Temperature_gas': r'$Temperature_{\mathrm{gas}}$',
    'CH2S': r'$C_{\mathrm{H_{2}S}}$',
    'Rotation_Speed': r'$Rotation\ Speed$'
}
# 按照自定义顺序创建标签列表
latex_labels_ordered = [latex_label_map[col] for col in custom_order]


# 5. 生成图表
plt.figure(figsize=(8, 6))

# 使用 shap.summary_plot 绘制图表
# 传入排序后的 SHAP 值、排序后的模拟特征值和排序后的 LaTeX 标签
shap.summary_plot(
    shap_df_ordered.values,
    features=mock_features_ordered,
    feature_names=latex_labels_ordered,
    show=False,
    sort=False  # 关键：禁用 SHAP 库的自动排序功能
)

# 6. 调整并显示最终图表
plt.xlabel("SHAP value (impact on model output)")
plt.tight_layout()
plt.show()


