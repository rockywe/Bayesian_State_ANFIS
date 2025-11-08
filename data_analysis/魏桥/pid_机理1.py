import math
import matplotlib.pyplot as plt
from dataclasses import dataclass


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
        # 计算时间差
        dt = current_time - self.last_time if self.last_time is not None else 0
        self.last_time = current_time

        # 初始化处理
        if dt <= 0:
            return 0.0

        error = self.config.setpoint - current_value

        # 积分项抗饱和
        new_integral = self.integral + error * dt
        derivative = (error - self.prev_error) / dt

        # 计算输出
        output = (
                self.config.Kp * error +
                self.config.Ki * new_integral +
                self.config.Kd * derivative
        )

        # 输出限幅
        min_limit, max_limit = self.config.output_limits
        if output > max_limit:
            output = max_limit
            new_integral = self.integral  # 停止积分
        elif output < min_limit:
            output = min_limit
            new_integral = self.integral  # 停止积分

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
        # 初始工艺参数
        self.process_params = {
            "煤气进口流量_m3h": 800,
            "进口煤气温度_C": 40,
            "进口煤气压力_kPa": 20,
            "脱硫液流量_m3h": 20,
            "脱硫液温度_C": 35,
            "脱硫液压力_kPa": 40,
            "转速_RPM": 1200,
            "进口H2S浓度_ppm": 1000,
            "L_exponent": 0.8,
            "enhancement_factor": 3.0
        }

        # PID控制器配置
        self.controllers = {
            "脱硫液流量": PIDController(PIDConfig(
                Kp=-0.05, Ki=-0.005, Kd=-0.01,
                setpoint=50,
                output_limits=(-5, 5)
            )),
            "转速": PIDController(PIDConfig(
                Kp=-2, Ki=-0.1, Kd=-0.5,
                setpoint=50,
                output_limits=(-50, 50)
            )),
            "煤气流量": PIDController(PIDConfig(
                Kp=0.03, Ki=0.001, Kd=0.005,
                setpoint=50,
                output_limits=(-10, 10)
            ))
        }

        # 历史数据记录
        self.history = {
            '时间': [],
            '浓度': [],
            '脱硫液流量': [],
            '转速': [],
            '煤气流量': []
        }

    def run_simulation(self, duration=600, dt=5):
        for t in range(0, duration + dt, dt):
            # 获取当前浓度
            current_conc = calculate_outlet_h2s(**self.process_params)

            # 记录数据
            self._record_data(t, current_conc)

            # 控制计算
            self._adjust_parameters(t)

            # 模拟扰动（测试控制器鲁棒性）
            if 150 < t < 300:
                self.process_params["进口H2S浓度_ppm"] = 1200
            else:
                self.process_params["进口H2S浓度_ppm"] = 1000

        self._visualize()

    def _record_data(self, t, conc):
        self.history['时间'].append(t)
        self.history['浓度'].append(conc)
        self.history['脱硫液流量'].append(self.process_params["脱硫液流量_m3h"])
        self.history['转速'].append(self.process_params["转速_RPM"])
        self.history['煤气流量'].append(self.process_params["煤气进口流量_m3h"])

    def _adjust_parameters(self, t):
        current_conc = self.history['浓度'][-1]

        # 各控制器独立调整
        adj_L = self.controllers["脱硫液流量"].compute(current_conc, t)
        adj_RPM = self.controllers["转速"].compute(current_conc, t)
        adj_G = self.controllers["煤气流量"].compute(current_conc, t)

        # 应用调整
        self.process_params["脱硫液流量_m3h"] = max(0, min(
            self.process_params["脱硫液流量_m3h"] + adj_L, 100))
        self.process_params["转速_RPM"] = max(800, min(
            self.process_params["转速_RPM"] + adj_RPM, 2000))
        self.process_params["煤气进口流量_m3h"] = max(500, min(
            self.process_params["煤气进口流量_m3h"] + adj_G, 1500))

    def _visualize(self):
        plt.figure(figsize=(14, 12))
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'font.family': 'DejaVu Sans',
            'grid.color': '#dddddd'
        })

        # 主浓度控制图
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(self.history['时间'], self.history['浓度'],
                 color='#2c3e50', linewidth=2, alpha=0.9, label='实际浓度')
        ax1.axhline(50, color='#e74c3c', linestyle='--', linewidth=1.5, label='目标浓度')
        ax1.set_title('H₂S浓度控制效果', pad=15)
        ax1.set_ylabel('浓度 (ppm)', labelpad=10)
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.4)
        ax1.set_facecolor('#f8f9fa')

        # 脱硫液流量控制图
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(self.history['时间'], self.history['脱硫液流量'],
                 color='#27ae60', linewidth=2, alpha=0.8,
                 marker='o', markersize=4, markevery=8)
        ax2.set_title('脱硫液流量控制', pad=10)
        ax2.set_ylabel('流量 (m³/h)', labelpad=10)
        ax2.grid(True, alpha=0.4)
        ax2.set_facecolor('#f8f9fa')
        ax2.set_ylim(0, 100)

        # 转速控制图
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(self.history['时间'], self.history['转速'],
                 color='#2980b9', linewidth=2, alpha=0.8,
                 marker='s', markersize=4, markevery=8)
        ax3.set_title('转速控制', pad=10)
        ax3.set_ylabel('转速 (RPM)', labelpad=10)
        ax3.grid(True, alpha=0.4)
        ax3.set_facecolor('#f8f9fa')
        ax3.set_ylim(800, 2000)

        # 煤气流量控制图
        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(self.history['时间'], self.history['煤气流量'],
                 color='#8e44ad', linewidth=2, alpha=0.8,
                 marker='^', markersize=4, markevery=8)
        ax4.set_title('煤气流量控制', pad=10)
        ax4.set_xlabel('时间 (秒)', labelpad=10)
        ax4.set_ylabel('流量 (m³/h)', labelpad=10)
        ax4.grid(True, alpha=0.4)
        ax4.set_facecolor('#f8f9fa')
        ax4.set_ylim(500, 1500)

        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(hspace=0.35)
        plt.show()


# ========================
# 执行程序
# ========================
if __name__ == "__main__":
    # 单点测试
    test_params = {
        "煤气进口流量_m3h": 800,
        "进口煤气温度_C": 40,
        "进口煤气压力_kPa": 20,
        "脱硫液流量_m3h": 20,
        "脱硫液温度_C": 35,
        "脱硫液压力_kPa": 40,
        "转速_RPM": 1200,
        "进口H2S浓度_ppm": 1000
    }
    print("初始工况测试:", calculate_outlet_h2s(**test_params), "ppm")

    # 运行控制模拟
    system = ControlSystem()
    system.run_simulation(duration=600)