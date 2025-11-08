import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib as mpl

# ========================
# 全局字体设置
# ========================
mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False


# ========================
# 核心工艺模型
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
        L_exponent=0.2279,
        RPM_exponent=0.5448,
        enhancement_factor=31,
        pH=12.0,  # 新增pH参数
        ch2o2=0.3  # 新增H2O2浓度参数
):
    # 物理常数
    D_H2S = 1.8e-9  # H2S扩散系数(m²/s)
    R_gas = 8.314  # 气体常数(J/mol/K)

    # 设备参数（标准化）
    R_inner = 0.042  # 转子内径(m)
    R_outer = 0.146  # 转子外径(m)
    h_packing = 0.02  # 填料高度(m)
    alpha = 500  # 填料比表面积(m²/m³)

    try:
        # 单位转换
        G_m3s = 煤气进口流量_m3h / 3600
        T_gas = 进口煤气温度_C + 273.15

        L_m3s = 脱硫液流量_m3h / 3600
        cross_area = math.pi * (R_outer ** 2 - R_inner ** 2)

        # 液膜更新模型
        L = L_m3s / (alpha * cross_area * h_packing) if cross_area != 0 else 0
        R_avg = math.sqrt(R_inner * R_outer)

        t_new = (R_outer - R_inner) / (
                0.02107 * (L ** L_exponent) *
                ((R_avg * 转速_RPM) ** RPM_exponent) *
                enhancement_factor
        )

        # 反应动力学计算
        al = 10 ** (11.73 - pH)
        a = 1 / (1 + al)
        ooh = ch2o2 * a
        k_appi = 103.4 * ooh  # 表观速率常数

        # 液膜传质系数
        if t_new > 0 and k_appi > 0:
            sqrt_k_appi_t = math.sqrt(k_appi * t_new)
            term1 = t_new * math.erf(sqrt_k_appi_t)
            term2 = math.sqrt(t_new / (math.pi * k_appi)) * math.exp(-k_appi * t_new)
            term3 = 0.5 / k_appi * math.erf(sqrt_k_appi_t)
            kl1 = (math.sqrt(D_H2S * k_appi) / t_new) * (term1 + term2 + term3)
        else:
            kl1 = 0

        # 总传质系数
        ky_cal = (R_gas * T_gas) * kl1

        # 物料平衡
        y_in = 进口H2S浓度_ppm * 1e-6
        a1 = (1 - y_in) / y_in
        a2 = alpha * math.pi * h_packing * (R_outer ** 2 - R_inner ** 2) / G_m3s if G_m3s > 0 else 0

        y_out = 1 / (a1 * math.exp(a2 * ky_cal) + 1e-9)  # 防止除零
        outlet_ppm = y_out * 1e6

        return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm * 1.2))
    except Exception as e:
        return 进口H2S浓度_ppm


# ========================
# 控制算法类
# ========================
@dataclass
class PIDConfig:
    Kp: float
    Ki: float
    Kd: float
    output_limits: tuple


class PIDController:
    def __init__(self, config: PIDConfig, setpoint: float):
        self.config = config
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

    def compute(self, current_value, current_time):
        dt = current_time - self.last_time if self.last_time else 0.1
        dt = max(dt, 0.1)  # 最小时间步长

        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        # 抗积分饱和
        raw_output = (
                self.config.Kp * error +
                self.config.Ki * self.integral +
                self.config.Kd * derivative
        )

        # 输出限幅
        output = max(self.config.output_limits[0],
                     min(raw_output, self.config.output_limits[1]))

        # 积分分离
        if abs(error) > self.setpoint * 0.2:
            output = self.config.Kp * error + self.config.Kd * derivative

        self.prev_error = error
        self.last_time = current_time
        return output


class FuzzyController:
    def __init__(self, setpoint, output_limits):
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.prev_error = 0.0
        self.last_time = None

        # 模糊集定义
        self.e_mf = {
            'NB': lambda e: self._trimf(e, -100, -100, -30),
            'NS': lambda e: self._trimf(e, -60, -30, 0),
            'Z': lambda e: self._trimf(e, -10, 0, 10),
            'PS': lambda e: self._trimf(e, 0, 30, 60),
            'PB': lambda e: self._trimf(e, 30, 100, 100)
        }

        self.de_mf = {
            'NB': lambda de: self._trimf(de, -30, -30, -10),
            'NS': lambda de: self._trimf(de, -20, -10, 0),
            'Z': lambda de: self._trimf(de, -3, 0, 3),
            'PS': lambda de: self._trimf(de, 0, 10, 20),
            'PB': lambda de: self._trimf(de, 10, 30, 30)
        }

        self.output_mf = {'NB': -90, 'NS': -40, 'Z': 0, 'PS': 40, 'PB': 90}

        self.rules = [
            ['NB', 'NB', 'PB'], ['NB', 'NS', 'PB'], ['NB', 'Z', 'PB'],
            ['NB', 'PS', 'PS'], ['NB', 'PB', 'Z'],
            ['NS', 'NB', 'PB'], ['NS', 'NS', 'PS'], ['NS', 'Z', 'PS'],
            ['NS', 'PS', 'Z'], ['NS', 'PB', 'NS'],
            ['Z', 'NB', 'PS'], ['Z', 'NS', 'Z'], ['Z', 'Z', 'Z'],
            ['Z', 'PS', 'Z'], ['Z', 'PB', 'NS'],
            ['PS', 'NB', 'NS'], ['PS', 'NS', 'Z'], ['PS', 'Z', 'NS'],
            ['PS', 'PS', 'NB'], ['PS', 'PB', 'NB'],
            ['PB', 'NB', 'Z'], ['PB', 'NS', 'NB'], ['PB', 'Z', 'NB'],
            ['PB', 'PS', 'NB'], ['PB', 'PB', 'NB']
        ]

    def _trimf(self, x, a, b, c):
        return max(0, min((x - a) / (b - a), (c - x) / (c - b))) if b != c else 1.0

    def compute(self, current_value, current_time):
        dt = current_time - self.last_time if self.last_time else 0.1
        error = self.setpoint - current_value
        de = (error - self.prev_error) / dt if dt > 0 else 0.0

        # 模糊化
        e_degree = {k: mf(error) for k, mf in self.e_mf.items()}
        de_degree = {k: mf(de) for k, mf in self.de_mf.items()}

        # 推理与去模糊化
        output_strength = {}
        for rule in self.rules:
            e_term, de_term, out_term = rule
            strength = min(e_degree[e_term], de_degree[de_term])
            if out_term in output_strength:
                output_strength[out_term] = max(output_strength[out_term], strength)
            else:
                output_strength[out_term] = strength

        # 加权平均去模糊化
        numerator = sum(strength * self.output_mf[term] for term, strength in output_strength.items())
        denominator = sum(output_strength.values())
        output = numerator / denominator if denominator else 0

        # 输出限幅
        output = max(self.output_limits[0], min(output, self.output_limits[1]))

        self.prev_error = error
        self.last_time = current_time
        return output * 0.8  # 增益调整


# ========================
# 控制系统主类
# ========================
class ControlSystem:
    def __init__(self):
        self.initial_params = {
            "煤气进口流量_m3h": 1500,
            "进口煤气温度_C": 40,
            "进口煤气压力_kPa": 20,
            "脱硫液流量_m3h": 30,
            "脱硫液温度_C": 35,
            "脱硫液压力_kPa": 40,
            "转速_RPM": 450,
            "进口H2S浓度_ppm": 1000,
            "pH": 12.0,  # 新增参数
            "ch2o2": 0.3  # 新增参数
        }
        self.target_h2s = 20.0

        # 初始化控制器
        self.pid_L = PIDController(PIDConfig(-0.25, -0.001, -0.015, (-1.5, 1.5)), self.target_h2s)
        self.pid_G = PIDController(PIDConfig(0.01, 0.0003, 0.002, (-10, 10)), self.target_h2s)
        self.pid_RPM = PIDController(PIDConfig(-0.8, -0.03, -0.6, (-30, 30)), self.target_h2s)
        self.fuzzy_RPM = FuzzyController(self.target_h2s, (-50, 50))

        # 历史数据存储
        self.history_pid = self._init_history()
        self.history_fuzzy = self._init_history()

    def _init_history(self):
        return {
            '时间': [], '浓度': [], '脱硫液流量': [],
            '转速': [], '煤气流量': [], 'RPM_adj': [], 'Error': []
        }

    def run_simulation(self, duration=600, dt=5, control_type='PID'):
        params = self.initial_params.copy()
        history = self.history_pid if control_type == 'PID' else self.history_fuzzy
        history.clear()

        for t in range(0, duration + dt, dt):
            # 扰动模拟
            params["进口H2S浓度_ppm"] = 1100 if 200 < t < 400 else 1000

            current_conc = calculate_outlet_h2s(**params)
            error = self.target_h2s - current_conc

            # 记录数据
            history['时间'].append(t)
            history['浓度'].append(current_conc)
            history['Error'].append(error)
            history['脱硫液流量'].append(params["脱硫液流量_m3h"])
            history['转速'].append(params["转速_RPM"])
            history['煤气流量'].append(params["煤气进口流量_m3h"])

            # 控制调整
            adj_L = self.pid_L.compute(current_conc, t)
            adj_G = self.pid_G.compute(current_conc, t)

            if control_type == 'PID':
                adj_RPM = self.pid_RPM.compute(current_conc, t)
            else:
                adj_RPM = self.fuzzy_RPM.compute(current_conc, t)

            # 应用控制量
            params["脱硫液流量_m3h"] = max(0, min(params["脱硫液流量_m3h"] + adj_L, 60))
            params["转速_RPM"] = max(0, min(params["转速_RPM"] + adj_RPM, 900))
            params["煤气进口流量_m3h"] = max(0, min(params["煤气进口流量_m3h"] + adj_G, 3000))

            history['RPM_adj'].append(adj_RPM)

    def visualize(self):
        plt.figure(figsize=(14, 10))

        # 浓度对比
        plt.subplot(2, 2, 1)
        plt.plot(self.history_pid['时间'], self.history_pid['浓度'], label='PID')
        plt.plot(self.history_fuzzy['时间'], self.history_fuzzy['浓度'], label='Fuzzy')
        plt.axhline(self.target_h2s, linestyle='--', color='gray')
        plt.title('出口H2S浓度对比')
        plt.legend()

        # 转速调整
        plt.subplot(2, 2, 2)
        plt.plot(self.history_pid['时间'], self.history_pid['RPM_adj'], label='PID调整量')
        plt.plot(self.history_fuzzy['时间'], self.history_fuzzy['RPM_adj'], label='Fuzzy调整量')
        plt.title('转速控制量对比')
        plt.legend()

        # 操作参数
        plt.subplot(2, 2, 3)
        plt.plot(self.history_pid['时间'], self.history_pid['脱硫液流量'], label='PID液量')
        plt.plot(self.history_fuzzy['时间'], self.history_fuzzy['脱硫液流量'], label='Fuzzy液量')
        plt.title('脱硫液流量对比')
        plt.legend()

        # 误差分析
        plt.subplot(2, 2, 4)
        plt.plot(self.history_pid['时间'], self.history_pid['Error'], label='PID误差')
        plt.plot(self.history_fuzzy['时间'], self.history_fuzzy['Error'], label='Fuzzy误差')
        plt.title('控制误差对比')
        plt.legend()

        plt.tight_layout()
        plt.savefig('./control_comparison.png', dpi=300)
        plt.show()


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    system = ControlSystem()
    system.run_simulation(control_type='PID')
    system.run_simulation(control_type='Fuzzy')
    system.visualize()