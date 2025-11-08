import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Load the CSV file with specified encoding to handle potential encoding issues
file_path_csv = '../硫化氢 - 处理.csv'
data_csv = pd.read_csv(file_path_csv, encoding='gbk')

# Filter the data for different types
unique_types = data_csv['种类'].unique()

# Set up a font that supports Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # Using SimHei for Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Ensure that the negative sign is displayed correctly

# Set up the matplotlib figure for H2S removal
plt.figure(figsize=(14, 7))

# Plot H2S脱除率 for different types
for item in unique_types:
    subset = data_csv[data_csv['种类'] == item]
    sns.lineplot(x=subset['w'], y=subset['H2S脱除率'], label=f'H2S脱除率 - {item}', marker='o', linewidth=2.5)

plt.title('H2S脱除率 vs w for different 种类')
plt.xlabel('w (rpm)')
plt.ylabel('H2S脱除率 (%)')
plt.legend()
plt.grid(True)
# plt.savefig('./plot_H2S_removal.png')
plt.show()

# Set up the matplotlib figure for CO2 removal
plt.figure(figsize=(14, 7))

# Plot CO2脱除率 for different types
for item in unique_types:
    subset = data_csv[data_csv['种类'] == item]
    sns.lineplot(x=subset['w'], y=subset['CO2脱除率'], label=f'CO2脱除率 - {item}', marker='o', linewidth=2.5)

plt.title('CO2脱除率 vs w for different 种类')
plt.xlabel('w (rpm)')
plt.ylabel('CO2脱除率 (%)')
plt.legend()
plt.grid(True)
# plt.savefig('./plot_CO2_removal.png')
plt.show()
