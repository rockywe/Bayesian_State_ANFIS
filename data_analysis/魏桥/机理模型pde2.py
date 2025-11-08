import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fipy import Grid1D, CellVariable, TransientTerm, DiffusionTerm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm  # 需要安装tqdm库

# ========================
# 核心计算函数（PDE版本）
# ========================
def calculate_outlet_h2s(
        煤气进口流量_m3h,
        进口煤气温度_C,
        进口煤气压力_kPa,
        脱硫液流量_m3h,
        脱硫液温度_C,
        脱硫液压力_kPa,
        转速_RPM,
        进口H2S浓度_ppm,
        # PDE参数
        nx=50,  # 空间网格数
        nt=100,  # 时间步数
        # 增强参数
        L_exponent=0.6,
        RPM_exponent=0.8,
        G_exponent=-0.25,
        gas_velocity_factor=1.2,
        enhancement_factor=2.5,
        contact_time_base=0.8
):
    # 物理常数
    D_H2S = 1.8e-9  # 基础扩散系数 (m²/s)
    H_H2S = 483.0  # 亨利常数 (atm·m³/mol)
    R_gas = 8.314  # 气体常数 (J/mol/K)
    liquid_density = 1100  # 脱硫液密度 (kg/m³)

    # 设备参数
    R_inner = 0.015  # 转子内径 (m)
    R_outer = 0.85  # 转子外径 (m)
    h_packing = 0.033  # 填料高度 (m)

    # 单位转换
    G_m3s = 煤气进口流量_m3h / 3600
    P_total = (进口煤气压力_kPa * 1000) / 101325  # kPa -> atm
    y_in = 进口H2S浓度_ppm * 1e-6

    L_m3s = 脱硫液流量_m3h / 3600
    T_liquid = 脱硫液温度_C + 273.15

    # 离心参数计算
    R_avg = math.sqrt(R_inner * R_outer)
    omega = 转速_RPM * 2 * math.pi / 60
    centrifugal_g = omega ** 2 * R_avg  # 离心加速度 (m/s²)

    # 液膜动力学参数
    u0 = 0.02107 * (L_m3s) ** 0.2279 * centrifugal_g ** 0.5  # 表面流速 (m/s)
    delta = 2.4e-5 * centrifugal_g **-0.62 * u0 ** 0.47  # 液膜厚度 (m)
    ts = delta ** 2 / (D_H2S * (1 + enhancement_factor))  # 特征时间 (s)

    # 增强扩散系数
    D_eff = (D_H2S * enhancement_factor
             * (centrifugal_g / 9.81) ** RPM_exponent
             * (L_m3s ** L_exponent)
             * (G_m3s ** G_exponent))

    # 反应速率常数（假设一级反应）
    k_reaction = 0.15 * enhancement_factor * (L_m3s ** 0.8)

    def solve_pde():
        # 网格系统
        mesh = Grid1D(dx=delta / nx, nx=nx)

        # 定义浓度变量
        c = CellVariable(name="H2S Concentration", mesh=mesh, value=0.0, hasOld=True)

        # 计算气液界面平衡浓度
        C_interface = (y_in * P_total) / (H_H2S * R_gas * T_liquid)

        # 边界条件
        c.constrain(C_interface, mesh.facesLeft)  # 左侧边界（气液界面）
        c.faceGrad.constrain(0, mesh.facesRight)  # 右侧边界（零梯度）

        # 控制方程：扩散 + 反应
        eq = TransientTerm() == DiffusionTerm(coeff=D_eff) - k_reaction * c

        # 时间步进参数
        dt = ts / nt

        # 时间迭代
        for _ in range(nt):
            if _ % 10 == 0:  # 每10步打印一次
                print(f"正在计算: 进度 {_ / nt * 100:.1f}%")
            c.updateOld()
            eq.solve(var=c, dt=dt)

        return c.faceValue[-1]  # 返回出口浓度

    try:
        # 求解PDE
        c_out = solve_pde()

        # 转换为气相ppm浓度
        y_out = (c_out * H_H2S * R_gas * T_liquid) / P_total
        outlet_ppm = y_out * 1e6

        # 结果限制
        return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm * 1.2))
    except Exception as e:
        print(f"计算错误: {str(e)}")
        return 0.0


# ========================
# 模型评估类（保持不变）
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

        print(f"⏳ 开始读取文件: {self.file_path}")
        self.df = pd.read_excel(self.file_path)
        print(f"✅ 成功读取 {len(self.df)} 行数据")

        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列：{missing_cols}")

        self.df = self.df.dropna(subset=required_columns)

    def predict(self):
        print("\n🔍 开始进行预测计算...")
        tqdm.pandas()  # 激活pandas进度条

        def _process_row(row):
            """带进度提示的单行处理"""
            current_index = row.name + 1  # 从1开始计数
            if current_index % 10 == 0:
                print(f"  正在处理第 {current_index}/{len(self.df)} 行", end='\r', flush=True)

            try:
                return calculate_outlet_h2s(
                    煤气进口流量_m3h=row["煤气进口流量"],
                    进口煤气温度_C=row["进口煤气温度"],
                    进口煤气压力_kPa=row["进口煤气压力"],
                    脱硫液流量_m3h=row["脱硫液流量"],
                    脱硫液温度_C=row["脱硫液温度"],
                    脱硫液压力_kPa=row["脱硫液压力"],
                    转速_RPM=row["转速"],
                    进口H2S浓度_ppm=row["进口H2S浓度"]
                )
            except Exception as e:
                print(f"\n⚠️ 第 {current_index} 行计算错误: {str(e)}")
                return 0.0

        # 使用进度条包装apply
        self.df["预测H2S浓度_ppm"] = self.df.progress_apply(_process_row, axis=1)
        print("\n🎉 预测完成！")

    def evaluate(self):
        y_true = self.df["出口H2S浓度"]
        y_pred = self.df["预测H2S浓度_ppm"]

        self.metrics = {
            "R²": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

    def visualize(self):
        plt.figure(figsize=(10, 4))

        # 预测 vs 实际
        plt.subplot(121)
        plt.scatter(self.df["出口H2S浓度"], self.df["预测H2S浓度_ppm"], alpha=0.6)
        plt.plot([0, 2000], [0, 2000], 'r--')
        plt.xlabel("实际浓度 (ppm)")
        plt.ylabel("预测浓度 (ppm)")
        plt.title(f"预测结果 (R²={self.metrics['R²']:.2f})")

        # 误差分布
        plt.subplot(122)
        errors = self.df["出口H2S浓度"] - self.df["预测H2S浓度_ppm"]
        plt.hist(errors, bins=30, alpha=0.7)
        plt.xlabel("预测误差 (ppm)")
        plt.ylabel("频数")
        plt.title("误差分布")

        plt.tight_layout()
        plt.show()


# ========================
# 执行主程序
# ========================
if __name__ == "__main__":
    evaluator = ModelEvaluator("D:\coding\data_higee_sxh\data_analysis\魏桥\脱硫数据整理.xlsx")
    evaluator.load_data()
    evaluator.predict()
    evaluator.evaluate()

    print("\n模型评估结果:")
    print(f"R²: {evaluator.metrics['R²']:.3f}")
    print(f"MAE: {evaluator.metrics['MAE']:.1f} ppm")
    print(f"RMSE: {evaluator.metrics['RMSE']:.1f} ppm")
    print(f"MAPE: {evaluator.metrics['MAPE']:.1f}%")

    evaluator.visualize()