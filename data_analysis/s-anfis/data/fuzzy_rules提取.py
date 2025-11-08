# 导入所需的库
import json
import pandas as pd

# --- 文件路径配置 ---
# 输入的 JSON 文件名
json_file_path = 'fuzzy_rules.json'
# 输出的 Excel 文件名 (建议使用新名称以区分)
excel_file_path = 'fuzzy_rules_extracted_en.xlsx'

# --- 翻译映射 ---
# 创建一个字典来映射中文到英文
translation_map = {
    '低': 'Low',
    '中': 'Middle',
    '高': 'High'
}

# --- 主逻辑 ---
try:
    # 1. 读取并解析 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 准备一个空列表来存储提取的数据
    extracted_data = []

    # 3. 遍历 JSON 数据中的每一条规则
    for rule in data:
        # 安全地提取数据
        rule_number = rule.get('rule_id')
        if_condition = rule.get('condition')
        then_equation = rule.get('consequent_equation')

        # --- 新增：翻译 IF 条件中的文本 ---
        # 确保 if_condition 是一个字符串，以防数据中存在空值
        if isinstance(if_condition, str):
            # 遍历翻译映射字典，并替换所有匹配项
            for chinese, english in translation_map.items():
                if_condition = if_condition.replace(chinese, english)

        # 将提取并处理后的数据追加到列表中
        extracted_data.append([rule_number, if_condition, then_equation])

    # 4. 定义 Excel 的列名
    columns = ['规则数', 'IF', 'THEN']

    # 5. 使用 pandas 创建一个 DataFrame
    df = pd.DataFrame(extracted_data, columns=columns)

    # 6. 将 DataFrame 保存为 Excel 文件
    df.to_excel(excel_file_path, index=False)

    print(f"✅ 数据提取成功，并将'低/中/高'替换为'Low/Medium/High'。")
    print(f"文件已保存为: {excel_file_path}")

except FileNotFoundError:
    print(f"❌ 错误：找不到文件 '{json_file_path}'。请确保文件与脚本在同一目录下。")
except Exception as e:
    print(f"❌ 处理过程中发生错误: {e}")
