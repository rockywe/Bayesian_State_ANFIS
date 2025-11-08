import pandas as pd
from scipy.stats import wilcoxon

# Load the Excel file
file_path = '12.24 - 分析.xlsx'
excel_data = pd.ExcelFile(file_path)

# Load data from 'Sheet4'
sheet4_data = excel_data.parse('Sheet4')

# Clean the data: remove rows with NaN or infinite values
clean_sheet4_data = sheet4_data.replace([float('inf'), float('-inf')], pd.NA).dropna()

# Specify the columns to compare
columns_to_compare = ['300组增量', '400组增量', 'CK组增量']

# Perform pairwise Wilcoxon signed-rank tests for significance
wilcoxon_results = {}
for i, col1 in enumerate(columns_to_compare):
    for j, col2 in enumerate(columns_to_compare):
        if i < j:  # Avoid duplicate comparisons
            stat, p_value = wilcoxon(clean_sheet4_data[col1], clean_sheet4_data[col2])
            wilcoxon_results[(col1, col2)] = {'Wilcoxon Statistic': stat, 'P-Value': p_value}

# Convert results to DataFrame for display
wilcoxon_df = pd.DataFrame.from_dict(wilcoxon_results, orient='index')

# Print results
print("Pairwise Wilcoxon Signed-Rank Test Results:")
print(wilcoxon_df)

