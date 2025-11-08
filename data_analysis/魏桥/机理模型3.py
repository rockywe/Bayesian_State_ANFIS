import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ========================
# 核心计算函数
# ========================
def calculate_outlet_h2s(
        煤气进口流量_m3h,  # m³/h (对应Excel列"煤气进口流量")
        进口煤气温度_C,  # ℃ (对应Excel列"进口煤气温度")
        进口煤气压力_kPa,  # kPa (对应Excel列"进口煤气压力")
        脱硫液流量_m3h,  # m³/h (对应Excel列"脱硫液流量")
        脱硫液温度_C,  # ℃ (对应Excel列"脱硫液温度")
        脱硫液压力_kPa,  # kPa (对应Excel列"脱硫液压力")
        转速_RPM,  # RPM (对应Excel列"转速")
        进口H2S浓度_ppm,  # ppm (对应Excel列"进口H2S浓度")
        # 增强参数
        L_exponent=0.6,
        RPM_exponent=0.8,
        G_exponent=-0.25,
        gas_velocity_factor=1.2,
        enhancement_factor=2.5,
        contact_time_base=0.8
):
    # 物理常数
    D_H2S = 1.8e-9  # H2S扩散系数 (m²/s)
    H_H2S = 483.0  # 亨利常数 (atm·m³/mol)
    alpha = 800  # 有效比表面积 (m²/m³)
    R_gas = 0.0821  # 气体常数 (L·atm/mol/K)
    liquid_density = 1100  # 脱硫液密度 (kg/m³)

    # 设备参数
    R_inner = 0.015  # 转子内径 (m)
    R_outer = 0.85  # 转子外径 (m)
    h_packing = 0.033  # 填料高度 (m)
    N_stages = 80  # 理论级数

    # 单位转换
    G_m3s = 煤气进口流量_m3h / 3600
    T_gas = 进口煤气温度_C + 273.15
    P_total = 进口煤气压力_kPa / 101.325
    y_in = 进口H2S浓度_ppm * 1e-6

    L_m3s = 脱硫液流量_m3h / 3600
    T_liquid = 脱硫液温度_C + 273.15

    R_avg = math.sqrt(R_inner * R_outer)
    omega = 转速_RPM * 2 * math.pi / 60

    # 增强传质模型
    def calculate_kL():
        centrifugal_g = (omega  ** 2 * R_avg) / 9.81
        kLa = 0.024 * enhancement_factor * (
            centrifugal_g  ** (RPM_exponent * enhancement_factor) *
              L_m3s  ** (L_exponent * enhancement_factor) *
                         G_m3s  ** (G_exponent * enhancement_factor)
        )
        return kLa / alpha

    # 动态物料平衡
    def mass_balance(kL):
        cross_area = math.pi * (R_outer  ** 2 - R_inner  ** 2)
        liquid_velocity = L_m3s / cross_area if cross_area != 0 else 0
        gas_velocity = G_m3s / cross_area if cross_area != 0 else 0

        combined_velocity = (liquid_velocity +
                             gas_velocity_factor * gas_velocity * enhancement_factor)
        residence_time = contact_time_base * h_packing / combined_velocity if combined_velocity != 0 else 0

        NTU = kL * alpha * residence_time
        y_out = y_in * math.exp(-NTU / (1 + NTU / (5 * enhancement_factor)))
        return y_out * 0.1  # 最终调整系数

    try:
        kL = calculate_kL()
        y_out = mass_balance(kL)
        outlet_ppm = y_out * 1e5
        return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm * 1.2))
    except:
        return 0.0


# ========================
# 模型评估类
# ========================
class ModelEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.results = None
        self.metrics = None

    def load_data(self):
        required_columns = [
            "煤气进口流量", "进口煤气温度", "进口煤气压力",
            "脱硫液流量", "脱硫液温度", "脱硫液压力",
            "转速", "进口H2S浓度", "出口H2S浓度"
        ]

        self.df = pd.read_excel(self.file_path)

        # 列名验证
        print("实际数据列:", self.df.columns.tolist())
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列：{missing_cols}")

        self.df = self.df.dropna(subset=required_columns)
        print(f"成功加载 {len(self.df)} 条有效数据")

    def predict(self):
        def apply_model(row):
            params = {
                "煤气进口流量_m3h": row["煤气进口流量"],
                "进口煤气温度_C": row["进口煤气温度"],
                "进口煤气压力_kPa": row["进口煤气压力"],
                "脱硫液流量_m3h": row["脱硫液流量"],
                "脱硫液温度_C": row["脱硫液温度"],
                "脱硫液压力_kPa": row["脱硫液压力"],
                "转速_RPM": row["转速"],
                "进口H2S浓度_ppm": row["进口H2S浓度"]
            }
            return calculate_outlet_h2s(**params)

        self.df["预测H2S浓度_ppm"] = self.df.apply(apply_model, axis=1)

    def evaluate(self):
        y_true = self.df["出口H2S浓度"]
        y_pred = self.df["预测H2S浓度_ppm"]

        self.metrics = {
            "R²": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

        self.results = self.df[[
            "出口H2S浓度", "预测H2S浓度_ppm"
        ]].copy()
        self.results["绝对误差"] = np.abs(self.results["出口H2S浓度"] - self.results["预测H2S浓度_ppm"])

    def visualize(self):
        """生成三个独立图表"""
        self._plot_scatter()
        self._plot_error_distribution()
        self._plot_error_trend()

    def _plot_scatter(self):
        """预测 vs 实际散点图"""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.results["出口H2S浓度"], self.results["预测H2S浓度_ppm"],
                    alpha=0.6, color='#3498db', edgecolor='w', s=80)

        max_val = self.results[["出口H2S浓度", "预测H2S浓度_ppm"]].max().max()
        plt.plot([0, max_val], [0, max_val], '--', color='#e74c3c', linewidth=2)

        plt.title(f"预测结果对比 (R² = {self.metrics['R²']:.3f})")
        plt.xlabel("实际浓度 (ppm)")
        plt.ylabel("预测浓度 (ppm)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_error_distribution(self):
        """误差分布直方图"""
        plt.figure(figsize=(8, 6))
        plt.hist(self.results["绝对误差"], bins=15,
                 color='#2ecc71', edgecolor='white', alpha=0.8)

        plt.title("绝对误差分布")
        plt.xlabel("绝对误差 (ppm)")
        plt.ylabel("频数")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_error_trend(self):
        """误差趋势图"""
        plt.figure(figsize=(8, 6))
        sorted_data = self.results.sort_values("出口H2S浓度")
        plt.plot(sorted_data["出口H2S浓度"], sorted_data["绝对误差"],
                 color='#9b59b6', linewidth=2, alpha=0.8)

        plt.fill_between(sorted_data["出口H2S浓度"], sorted_data["绝对误差"],
                         color='#9b59b6', alpha=0.1)

        plt.title("误差随浓度变化趋势")
        plt.xlabel("实际浓度 (ppm)")
        plt.ylabel("绝对误差 (ppm)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_results(self, output_file="预测结果.xlsx"):
        report = pd.DataFrame([self.metrics])
        with pd.ExcelWriter(output_file) as writer:
            self.results.to_excel(writer, sheet_name='详细数据', index=False)
            report.to_excel(writer, sheet_name='性能指标', index=False)
        print(f"结果已保存至 {output_file}")


# ========================
# 参数敏感性分析
# ========================
def parameter_sensitivity_analysis():
    base_case = {
        "煤气进口流量_m3h": 800,
        "进口煤气温度_C": 40,
        "进口煤气压力_kPa": 20,
        "脱硫液流量_m3h": 20,
        "脱硫液温度_C": 35,
        "脱硫液压力_kPa": 40,
        "转速_RPM": 1200,
        "进口H2S浓度_ppm": 1000
    }

    analysis_config = [
        {
            "title": "脱硫液流量敏感性分析",
            "param": "脱硫液流量",
            "range": (0, 100),
            "step": 10,
            "xlabel": "脱硫液流量 (m³/h)",
            "enhance_params": {"L_exponent": 0.8, "enhancement_factor": 3.0},
            "color": "#2ecc71"
        },
        {
            "title": "转速敏感性分析",
            "param": "转速",
            "range": (800, 2000),
            "step": 50,
            "xlabel": "转速 (RPM)",
            "enhance_params": {"RPM_exponent": 1.0, "enhancement_factor": 3.0},
            "color": "#e74c3c"
        },
        {
            "title": "煤气流量敏感性分析",
            "param": "煤气进口流量",
            "range": (500, 1500),
            "step": 50,
            "xlabel": "煤气流量 (m³/h)",
            "enhance_params": {"G_exponent": -0.4, "gas_velocity_factor": 1.5},
            "color": "#3498db"
        }
    ]

    plt.rcParams.update({
        'font.sans-serif': 'SimHei',
        'axes.unicode_minus': False,
        'figure.dpi': 120,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

    for config in analysis_config:
        plt.figure(figsize=(8, 5))
        x_values = []
        y_normal = []
        y_enhanced = []

        for value in range(config["range"][0], config["range"][1] + config["step"], config["step"]):
            current_params = base_case.copy()
            current_params[config["param"]] = value
            normal_result = calculate_outlet_h2s(**current_params)

            enhanced_params = current_params.copy()
            enhanced_params.update(config["enhance_params"])
            enhanced_result = calculate_outlet_h2s(**enhanced_params)

            x_values.append(value)
            y_normal.append(normal_result)
            y_enhanced.append(enhanced_result)

        plt.plot(x_values, y_normal, color=config["color"],
                 linestyle='-', marker='o', label='默认参数')
        plt.plot(x_values, y_enhanced, color=config["color"],
                 linestyle='--', marker='s', label='优化参数')

        plt.title(config["title"])
        plt.xlabel(config["xlabel"])
        plt.ylabel("出口H₂S浓度 (ppm)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    DATA_FILE = "D:/coding/data_higee_sxh/data_analysis/魏桥/脱硫数据整理.xlsx"

    try:
        evaluator = ModelEvaluator(DATA_FILE)
        evaluator.load_data()
        evaluator.predict()
        evaluator.evaluate()

        print("\n模型性能指标：")
        for metric, value in evaluator.metrics.items():
            print(f"{metric}: {value:.4f}")

        evaluator.visualize()
        evaluator.save_results()

        # 执行敏感性分析（取消注释运行）
        parameter_sensitivity_analysis()

    except Exception as e:
        print(f"错误: {str(e)}")
        print("请检查：")
        print("1. Excel文件路径是否正确")
        print("2. 列名是否与以下完全一致：")
        print(
            "   [煤气进口流量, 进口煤气温度, 进口煤气压力, 脱硫液流量, 脱硫液温度, 脱硫液压力, 转速, 进口H2S浓度, 出口H2S浓度]")
        print("3. 数值数据是否包含非数字字符")