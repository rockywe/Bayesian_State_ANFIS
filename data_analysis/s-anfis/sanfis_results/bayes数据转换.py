import json
import pandas as pd
import os

# 1. 定义文件路径
json_file_path = './bfs数据/bsanfis_optimization_result_20250822_100621.json'
# 从输入文件名生成输出文件名
base_name = os.path.splitext(os.path.basename(json_file_path))[0]
# 将 "result" 替换为 "summary" 以便区分，并设置格式为 .xls
output_file_name = base_name.replace("result", "summary") + '.xls'

# 2. 读取并解析JSON文件
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误：找不到文件 '{json_file_path}'。请确保文件与脚本在同一目录下，或提供正确的文件路径。")
    exit()

# 3. 提取所需数据
# 获取优化历史记录列表
optimization_history = data.get('optimization_history', [])

# 准备一个列表来存储每一行的数据
records_to_save = []

# 遍历历史记录中的每一次评估
for record in optimization_history:
    # 检查 'metrics' 键是否存在，避免出错
    if 'metrics' in record:
        records_to_save.append({
            '训练轮次': record.get('evaluation_id'),
            'r2': record['metrics'].get('r2'),
            'rmse': record['metrics'].get('rmse'),
            '总分': record.get('final_score')
        })

# 4. 创建DataFrame并保存为XLS文件
if records_to_save:
    # 将数据列表转换为Pandas DataFrame
    df = pd.DataFrame(records_to_save)

    # 将DataFrame保存为XLS文件
    # 注意：保存为 .xls 格式需要安装 'xlwt' 库 (pip install xlwt)
    try:
        df.to_excel(output_file_name, index=False, engine='xlwt')
        print(f"数据已成功提取并保存到文件：'{output_file_name}'")
    except ImportError:
        print("错误：需要安装 'xlwt' 库才能保存为 .xls 格式。")
        print("请运行 'pip install xlwt' 命令进行安装。")
    except Exception as e:
        print(f"保存文件时发生错误：{e}")
else:
    print("在JSON文件中没有找到可处理的 'optimization_history' 数据。")

