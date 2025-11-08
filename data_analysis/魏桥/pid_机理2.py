import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib as mpl

# 方法一：全局设置（优先推荐）
mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 简体中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
            new_integral = self.integral
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
        煤气进口流量_m3h,  # m³/h
        进口煤气温度_C,  # ℃
        进口煤气压力_kPa,  # kPa
        脱硫液流量_m3h,  # m³/h
        脱硫液温度_C,  # ℃
        脱硫液压力_kPa,  # kPa
        转速_RPM,  # RPM
        进口H2S浓度_ppm,  # ppm
        L_exponent=0.6,  # 脱硫液流量指数
        RPM_exponent=0.8,  # 转速影响指数
        G_exponent=-0.25,  # 煤气流量指数
        gas_velocity_factor=1.2,  # 气体流速因子
        enhancement_factor=2.5,  # 强化因子
        contact_time_base=0.8  # 接触时间基准
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
        outlet_ppm = y_out * 1e6 * 0.1  # 最终调整系数

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

        # 优化后的PID参数配置
        self.controllers = {
            "脱硫液流量": PIDController(PIDConfig(
                Kp=-0.25, Ki=-0.001, Kd=-0.015,
                setpoint=80,
                output_limits=(-1.5, 1.5)
            )),
            "转速": PIDController(PIDConfig(
                Kp=-0.8, Ki=-0.03, Kd=-0.6,
                setpoint=80,
                output_limits=(-20, 20)
            )),
            "煤气流量": PIDController(PIDConfig(
                Kp=0.01, Ki=0.0003, Kd=0.002,
                setpoint=80,
                output_limits=(-3, 3)
            ))
        }

        self.history = {
            '时间': [], '浓度': [],
            '脱硫液流量': [], '转速': [], '煤气流量': []
        }
        self.prev_adj = {'L': 0, 'RPM': 0, 'G': 0}

    def run_simulation(self, duration=600, dt=5):
        for t in range(0, duration + dt, dt):
            current_conc = calculate_outlet_h2s(**self.process_params)
            self._record_data(t, current_conc)
            self._adjust_parameters(t)
            # 模拟扰动
            self.process_params["进口H2S浓度_ppm"] = 1100 if 200 < t < 400 else 1000

        self._visualize()

    def _record_data(self, t, conc):
        self.history['时间'].append(t)
        self.history['浓度'].append(conc)
        self.history['脱硫液流量'].append(self.process_params["脱硫液流量_m3h"])
        self.history['转速'].append(self.process_params["转速_RPM"])
        self.history['煤气流量'].append(self.process_params["煤气进口流量_m3h"])

    def _adjust_parameters(self, t):
        current_conc = self.history['浓度'][-1]

        # 带滤波的PID计算
        adj_L = self.controllers["脱硫液流量"].compute(current_conc, t)
        adj_L = 0.2 * adj_L + 0.8 * self.prev_adj['L']

        adj_RPM = self.controllers["转速"].compute(current_conc, t)
        adj_RPM = 0.2 * adj_RPM + 0.8 * self.prev_adj['RPM']

        adj_G = self.controllers["煤气流量"].compute(current_conc, t)
        adj_G = 0.2 * adj_G + 0.8 * self.prev_adj['G']

        # 应用带约束的调整
        self.process_params["脱硫液流量_m3h"] = max(15, min(
            self.process_params["脱硫液流量_m3h"] + adj_L, 40))
        self.process_params["转速_RPM"] = max(1100, min(
            self.process_params["转速_RPM"] + adj_RPM, 1400))
        self.process_params["煤气进口流量_m3h"] = max(700, min(
            self.process_params["煤气进口流量_m3h"] + adj_G, 1000))

        self.prev_adj = {'L': adj_L, 'RPM': adj_RPM, 'G': adj_G}

    def _visualize(self):
        # 配置全局样式
        plt.style.use('ggplot')
        colors = ['#2c3e50', '#27ae60', '#2980b9', '#8e44ad']
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'figure.figsize': (10, 6),
            'figure.dpi': 100
        })

        # 生成四个独立图表
        self._plot_concentration(colors[0])
        self._plot_liquid_flow(colors[1])
        self._plot_rpm(colors[2])
        self._plot_gas_flow(colors[3])
        plt.show()

    def _plot_concentration(self, color):
        plt.figure(1)
        plt.plot(self.history['时间'], self.history['浓度'],
                 color=color, linewidth=2, alpha=0.9)
        # plt.axhline(80, color='#e74c3c', linestyle='--', linewidth=1.5)
        plt.title('H₂S浓度控制效果 (优化后PID参数)', pad=12)
        plt.xlabel('时间 (秒)')
        plt.ylabel('浓度 (ppm)')
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

    def _plot_liquid_flow(self, color):
        plt.figure(2)
        plt.plot(self.history['时间'], self.history['脱硫液流量'],
                 color=color, linewidth=2, alpha=0.8)
        plt.title('脱硫液流量控制', pad=12)
        plt.xlabel('时间 (秒)')
        plt.ylabel('流量 (m³/h)')
        plt.ylim(15, 50)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

    def _plot_rpm(self, color):
        plt.figure(3)
        plt.plot(self.history['时间'], self.history['转速'],
                 color=color, linewidth=2, alpha=0.8)
        plt.title('转速控制', pad=12)
        plt.xlabel('时间 (秒)')
        plt.ylabel('转速 (RPM)')
        plt.ylim(1200, 1500)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

    def _plot_gas_flow(self, color):
        plt.figure(4)
        plt.plot(self.history['时间'], self.history['煤气流量'],
                 color=color, linewidth=2, alpha=0.8)
        plt.title('煤气流量控制', pad=12)
        plt.xlabel('时间 (秒)')
        plt.ylabel('流量 (m³/h)')
        plt.ylim(500, 1000)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()


# ========================
# 执行程序
# ========================
if __name__ == "__main__":
    # 运行控制模拟
    system = ControlSystem()
    system.run_simulation(duration=600)