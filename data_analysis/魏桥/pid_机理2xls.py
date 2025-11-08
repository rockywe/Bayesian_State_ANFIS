import math
from dataclasses import dataclass
import pandas as pd
import openpyxl
from openpyxl.styles import Alignment


# ========================
# PID控制器类
# ========================
@dataclass
class PIDConfig:
    Kp: float
    Ki: float
    Kd: float
    setpoint: float
    output_limits: tuple


class PIDController:
    def __init__(self, config: PIDConfig):
        self.config = config
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

    def compute(self, current_value, current_time):
        dt = current_time - self.last_time if self.last_time is not None else 0
        self.last_time = current_time

        if dt <= 0:
            return 0.0

        error = self.config.setpoint - current_value
        new_integral = self.integral + error * dt
        derivative = (error - self.prev_error) / dt

        output = (
                self.config.Kp * error +
                self.config.Ki * new_integral +
                self.config.Kd * derivative
        )

        min_limit, max_limit = self.config.output_limits
        if output > max_limit:
            output = max_limit
            new_integral = self.integral  # 停止积分
        elif output < min_limit:
            output = min_limit
            new_integral = self.integral

        self.integral = new_integral
        self.prev_error = error
        return output


# ========================
# 核心工艺计算模型
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
        L_exponent=0.6,
        RPM_exponent=0.8,
        G_exponent=-0.25,
        gas_velocity_factor=1.2,
        enhancement_factor=2.5,
        contact_time_base=0.8
):
    # 物理常数
    D_H2S = 1.8e-9
    H_H2S = 483.0
    alpha = 800
    R_gas = 0.0821
    liquid_density = 1100

    # 设备参数
    R_inner = 0.015
    R_outer = 0.85
    h_packing = 0.033
    N_stages = 80

    try:
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
        centrifugal_g = (omega  ** 2 * R_avg) / 9.81
        kLa = 0.024 * enhancement_factor * (
            centrifugal_g  ** (RPM_exponent * enhancement_factor) *
              L_m3s  ** (L_exponent * enhancement_factor) *
                         G_m3s  ** (G_exponent * enhancement_factor)
        )
        kL = kLa / alpha

        # 动态物料平衡
        cross_area = math.pi * (R_outer  ** 2 - R_inner  ** 2)
        liquid_velocity = L_m3s / cross_area if cross_area != 0 else 0
        gas_velocity = G_m3s / cross_area if cross_area != 0 else 0

        combined_velocity = (liquid_velocity +
                             gas_velocity_factor * gas_velocity * enhancement_factor)
        residence_time = contact_time_base * h_packing / combined_velocity if combined_velocity != 0 else 0

        NTU = kL * alpha * residence_time
        y_out = y_in * math.exp(-NTU / (1 + NTU / (5 * enhancement_factor)))
        outlet_ppm = y_out * 1e6 * 0.1

        return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm * 1.2))
    except:
        return 0.0


# ========================
# 控制模拟主程序
# ========================
class ControlSystem:
    def __init__(self):
        self.process_params = {
            "煤气进口流量_m3h": 800,
            "进口煤气温度_C": 40,
            "进口煤气压力_kPa": 20,
            "脱硫液流量_m3h": 22,
            "脱硫液温度_C": 35,
            "脱硫液压力_kPa": 40,
            "转速_RPM": 1250,
            "进口H2S浓度_ppm": 1000,
            "L_exponent": 0.8,
            "enhancement_factor": 3.0
        }

        self.controllers = {
            "脱硫液流量": PIDController(PIDConfig(
                Kp=-0.25, Ki=-0.001, Kd=-0.015,
                setpoint=50,
                output_limits=(-1.5, 1.5)
            )),
            "转速": PIDController(PIDConfig(
                Kp=-0.8, Ki=-0.03, Kd=-0.6,
                setpoint=50,
                output_limits=(-20, 20)
            )),
            "煤气流量": PIDController(PIDConfig(
                Kp=0.01, Ki=0.0003, Kd=0.002,
                setpoint=50,
                output_limits=(-3, 3)
            ))
        }

        self.history = {
            '时间步': [],
            '时间(s)': [],
            '当前浓度(ppm)': [],
            '目标浓度(ppm)': [],
            '脱硫液流量(m³/h)': [],
            '转速(RPM)': [],
            '煤气流量(m³/h)': [],
            'PID调整量_脱硫液': [],
            'PID调整量_转速': [],
            'PID调整量_煤气': [],
            '进口H2S浓度(ppm)': []
        }

        self.export_path = "process_control_data.xlsx"
        self.step_counter = 0

    def run_simulation(self, duration=600, dt=5):
        for t in range(0, duration + dt, dt):
            self.step_counter += 1
            current_conc = calculate_outlet_h2s(**self.process_params)
            self._record_data(t, current_conc)
            self._adjust_parameters(t)
            self._apply_disturbance(t)

        self._export_data()
        print(f"模拟完成，数据已保存至: {self.export_path}")

    def _record_data(self, t, conc):
        """记录每个时间步的全部数据"""
        self.history['时间步'].append(self.step_counter)
        self.history['时间(s)'].append(t)
        self.history['当前浓度(ppm)'].append(conc)
        self.history['目标浓度(ppm)'].append(self.controllers["脱硫液流量"].config.setpoint)
        self.history['脱硫液流量(m³/h)'].append(self.process_params["脱硫液流量_m3h"])
        self.history['转速(RPM)'].append(self.process_params["转速_RPM"])
        self.history['煤气流量(m³/h)'].append(self.process_params["煤气进口流量_m3h"])
        self.history['进口H2S浓度(ppm)'].append(self.process_params["进口H2S浓度_ppm"])

    def _adjust_parameters(self, t):
        """带详细调整量记录的参数调整"""
        current_conc = self.history['当前浓度(ppm)'][-1]

        # 计算原始PID输出
        raw_adj_L = self.controllers["脱硫液流量"].compute(current_conc, t)
        raw_adj_RPM = self.controllers["转速"].compute(current_conc, t)
        raw_adj_G = self.controllers["煤气流量"].compute(current_conc, t)

        # 应用一阶低通滤波
        adj_L = 0.2 * raw_adj_L + 0.8 * self.history['PID调整量_脱硫液'][-1] if self.step_counter > 1 else raw_adj_L
        adj_RPM = 0.2 * raw_adj_RPM + 0.8 * self.history['PID调整量_转速'][-1] if self.step_counter > 1 else raw_adj_RPM
        adj_G = 0.2 * raw_adj_G + 0.8 * self.history['PID调整量_煤气'][-1] if self.step_counter > 1 else raw_adj_G

        # 记录调整量
        self.history['PID调整量_脱硫液'].append(adj_L)
        self.history['PID调整量_转速'].append(adj_RPM)
        self.history['PID调整量_煤气'].append(adj_G)

        # 应用约束调整
        self.process_params["脱硫液流量_m3h"] = max(15, min(
            self.process_params["脱硫液流量_m3h"] + adj_L, 40))
        self.process_params["转速_RPM"] = max(1100, min(
            self.process_params["转速_RPM"] + adj_RPM, 1400))
        self.process_params["煤气进口流量_m3h"] = max(700, min(
            self.process_params["煤气进口流量_m3h"] + adj_G, 1000))

    def _apply_disturbance(self, t):
        """应用扰动模型"""
        if 200 < t < 400:
            self.process_params["进口H2S浓度_ppm"] = 1100
        else:
            self.process_params["进口H2S浓度_ppm"] = 1000

    def _export_data(self):
        """专业级Excel导出"""
        df = pd.DataFrame(self.history)

        # 创建带格式的Excel文件
        with pd.ExcelWriter(self.export_path, engine='openpyxl') as writer:
            # 主数据表
            df.to_excel(writer, sheet_name='过程数据', index=False)

            # 参数说明表
            param_data = {
                "参数名称": [
                    "比例系数 (Kp)", "积分系数 (Ki)", "微分系数 (Kd)",
                    "脱硫液流量限幅", "转速限幅", "煤气流量限幅",
                    "时间步长", "总时长"
                ],
                "脱硫液流量控制器": [
                    self.controllers["脱硫液流量"].config.Kp,
                    self.controllers["脱硫液流量"].config.Ki,
                    self.controllers["脱硫液流量"].config.Kd,
                    str(self.controllers["脱硫液流量"].config.output_limits),
                    "", "", "", ""
                ],
                "转速控制器": [
                    self.controllers["转速"].config.Kp,
                    self.controllers["转速"].config.Ki,
                    self.controllers["转速"].config.Kd,
                    "",
                    str(self.controllers["转速"].config.output_limits),
                    "", "", ""
                ],
                "煤气流量控制器": [
                    self.controllers["煤气流量"].config.Kp,
                    self.controllers["煤气流量"].config.Ki,
                    self.controllers["煤气流量"].config.Kd,
                    "", "",
                    str(self.controllers["煤气流量"].config.output_limits),
                    "", ""
                ],
                "模拟参数": [
                    "", "", "", "", "", "",
                    f"{self.history['时间步'][-1]} steps",
                    f"{self.history['时间(s)'][-1]}秒"
                ]
            }
            pd.DataFrame(param_data).to_excel(writer, sheet_name='参数配置', index=False)

            # 格式优化
            self._format_excel(writer.book)

    def _format_excel(self, workbook):
        """专业格式设置"""
        # 主数据表格式
        data_sheet = workbook['过程数据']
        data_sheet.column_dimensions['A'].width = 10  # 时间步
        data_sheet.column_dimensions['B'].width = 12  # 时间(s)

        # 数值精度设置
        number_columns = {
            'C': "0.00",  # 当前浓度
            'D': "0.00",  # 目标浓度
            'E': "0.000",  # 脱硫液流量
            'F': "0",  # 转速
            'G': "0.00",  # 煤气流量
            'H': "0.0000",  # PID调整量
            'I': "0.0000",
            'J': "0.0000",
            'K': "0"  # 进口浓度
        }

        for col, fmt in number_columns.items():
            for cell in data_sheet[col]:
                cell.number_format = fmt
                cell.alignment = Alignment(horizontal='right')

        # 参数表格式
        param_sheet = workbook['参数配置']
        param_sheet.column_dimensions['A'].width = 22
        param_sheet.column_dimensions['B'].width = 18
        param_sheet.column_dimensions['C'].width = 18
        param_sheet.column_dimensions['D'].width = 18

        # 冻结首行
        data_sheet.freeze_panes = 'A2'
        param_sheet.freeze_panes = 'A2'


# ========================
# 执行程序
# ========================
if __name__ == "__main__":
    # 依赖检查
    try:
        import pandas
        import openpyxl
    except ImportError:
        print("运行前请安装依赖库：")
        print("pip install pandas openpyxl")
        exit(1)

    # 运行模拟
    simulator = ControlSystem()
    simulator.run_simulation(duration=600, dt=5)