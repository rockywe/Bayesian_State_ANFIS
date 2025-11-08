import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans

# 1. 加载数据
data_file = './脱硫数据整理.xlsx'  # 替换为您的数据文件路径
sheet_name = 'Sheet1'  # 替换为您的数据工作表名称
df = pd.read_excel(data_file, sheet_name=sheet_name)

# 2. 数据分析 (示例)
print("原始数据概览:")
print(df.head())
print("\n数据类型:")
print(df.info())
print("\n缺失值统计:")
print(df.isnull().sum())
print("\n描述性统计:")
print(df.describe())

# 3. 数据筛选
def filter_data(df, missing_value_threshold=0.5, zscore_threshold=3, cluster_method='kmeans', n_clusters=5, target_samples=300):
    """
    根据多种策略筛选数据，并尽量保留指定数量的样本。

    Args:
        df (pd.DataFrame): 原始数据。
        missing_value_threshold (float): 缺失值比例阈值，超过该阈值的样本将被删除。
        zscore_threshold (float): Z-score 阈值，用于识别异常值。
        cluster_method (str): 聚类方法，'kmeans' 或 'dbscan'。
        n_clusters (int): KMeans 聚类数。
        target_samples (int): 目标保留样本数。

    Returns:
        pd.DataFrame: 筛选后的数据。
    """
    print("\n开始数据筛选...")
    filtered_df = df.copy()  # 创建数据副本，避免修改原始数据
    initial_rows = len(filtered_df)

    # 3.1 基于数据质量的选择
    print("\n3.1 基于数据质量的选择:")
    # 3.1.1 缺失值过多的样本
    print("  - 处理缺失值过多的样本...")
    filtered_df = filtered_df.dropna(thresh=len(filtered_df.columns) * (1 - missing_value_threshold))
    print(f"    - 删除了 {initial_rows - len(filtered_df)} 行")
    initial_rows = len(filtered_df)

    # 3.1.2 异常值明显的样本 (使用 Z-score)
    print("  - 处理异常值明显的样本 (Z-score)...")
    numeric_df = filtered_df.select_dtypes(include=np.number).copy()  # 仅选择数值列, 并复制
    for col in numeric_df.columns:
        filtered_df = filtered_df[np.abs(zscore(filtered_df[col])) < zscore_threshold]
    print(f"    - 删除了 {initial_rows - len(filtered_df)} 行")
    initial_rows = len(filtered_df)

    # 3.2 基于数据代表性的选择
    print("\n3.2 基于数据代表性的选择:")
    # 3.2.1 重复样本删除
    print("  - 删除重复样本...")
    filtered_df = filtered_df.drop_duplicates()
    print(f"    - 删除了 {initial_rows - len(filtered_df)} 行")
    initial_rows = len(filtered_df)

    # 3.2.2 基于聚类的采样 (KMeans 或 DBSCAN)
    print("  - 基于聚类的采样...")
    if cluster_method == 'kmeans':
        print("  - 使用 KMeans 聚类...")
        numeric_df = filtered_df.select_dtypes(include=np.number).copy()  # 在聚类前重新选择
        # 调整聚类数以满足目标样本数
        if len(filtered_df) < target_samples:
            n_clusters = len(filtered_df)  # 每个样本作为一个簇
        else:
             n_clusters = min(n_clusters, len(filtered_df)) # 确保 n_clusters 不超过样本数
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # 显式设置 n_init
        clusters = kmeans.fit_predict(numeric_df)  # 使用数值列进行聚类
        filtered_df['cluster'] = clusters
        # 从每个簇中选择一个代表性样本 (例如，距离簇中心最近的样本)
        representative_samples = []
        for cluster_id in range(n_clusters):
            cluster_df = filtered_df[filtered_df['cluster'] == cluster_id]
            if not cluster_df.empty:  # 确保簇不为空
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.sqrt(((cluster_df[numeric_df.columns] - center) ** 2).sum(axis=1))
                closest_sample_index = distances.idxmin()
                representative_samples.append(filtered_df.loc[closest_sample_index])
        filtered_df = pd.DataFrame(representative_samples)
        filtered_df = filtered_df.drop(columns=['cluster'], errors='ignore')  # 删除聚类标签
        print(f"    - 选择了 {len(filtered_df)} 个代表性样本")

    elif cluster_method == 'dbscan':
        try:
            from sklearn.cluster import DBSCAN
            print("  - 使用 DBSCAN 聚类...")
            # DBSCAN 不需要指定簇的数量，它通过密度连接来寻找簇
            #  调整 eps 参数以获得合适的簇数
            eps = 0.3  # 初始值，可能需要根据数据调整
            min_samples = 10
            numeric_df = filtered_df.select_dtypes(include=np.number).copy()  # 在聚类前重新选择
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # 需要根据数据调整 eps 和 min_samples
            clusters = dbscan.fit_predict(numeric_df)
            filtered_df['cluster'] = clusters
            n_clusters = len(np.unique(clusters))
             # 调整 eps 直到簇的数量接近目标样本数
            while n_clusters > target_samples * 1.5:  # 允许一些偏差
                eps += 0.1  # 逐步增加 eps
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(numeric_df)
                filtered_df['cluster'] = clusters
                n_clusters = len(np.unique(clusters))
            representative_samples = []
            for cluster_id in np.unique(clusters):
                if cluster_id == -1:  # -1 表示噪声点，不选择
                    continue
                cluster_df = filtered_df[filtered_df['cluster'] == cluster_id]
                if not cluster_df.empty:
                    center = cluster_df[numeric_df.columns].mean().values  # 使用簇内样本均值作为中心
                    distances = np.sqrt(((cluster_df[numeric_df.columns] - center) ** 2).sum(axis=1))
                    closest_sample_index = distances.idxmin()
                    representative_samples.append(filtered_df.loc[closest_sample_index])
            filtered_df = pd.DataFrame(representative_samples)
            filtered_df = filtered_df.drop(columns=['cluster'], errors='ignore')
            print(f"   - 选择了 {len(filtered_df)} 个代表性样本")
        except ImportError:
            print("   - DBSCAN 不可用，请确保 scikit-learn 版本 >= 0.20")

    # 3.3 调整最终样本数
    print("\n3.3 调整最终样本数...")
    if len(filtered_df) > target_samples:
        filtered_df = filtered_df.sample(n=target_samples, random_state=42)
        print(f"   - 随机选择了 {target_samples} 个样本")
    elif len(filtered_df) < target_samples and len(df) >= target_samples:
        #  如果筛选后的数据太少，则从原始数据中随机抽取一些样本进行补充
        supplement_df = df.sample(n=target_samples - len(filtered_df), random_state=42)
        filtered_df = pd.concat([filtered_df, supplement_df], ignore_index=True)
        print(f"   - 从原始数据补充了 {target_samples - len(filtered_df)} 个样本，总数为 {len(filtered_df)}")
    print("数据筛选完成.")
    return filtered_df

# 4. 应用筛选并输出结果
filtered_df = filter_data(df, target_samples=100) # 指定目标样本数为 300
print("\n筛选后的数据概览:")
print(filtered_df.head())
print("\n筛选后的数据维度:")
print(filtered_df.shape)

# (可选) 保存筛选后的数据
filtered_df.to_excel('./filtered_data.xlsx', index=False)
print("\n筛选后的数据已保存到 ./filtered_data.xlsx")
