import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib as mpl

# 方法一：全局设置（优先推荐）
mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 简体中文，增加备选字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========================
# PID控制器类
# ========================
@dataclass
class PIDConfig:
    Kp: float
    Ki: float
    Kd: float
    # setpoint: float # Removed individual setpoint
    output_limits: tuple


class PIDController:
    def __init__(self, config: PIDConfig, setpoint: float):
        self.config = config
        self.setpoint = setpoint # Added setpoint here
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

    def compute(self, current_value, current_time):
        dt = current_time - self.last_time if self.last_time is not None else 0
        self.last_time = current_time

        if dt <= 0:
            # Handle the first call or zero time step
            self.prev_error = self.setpoint - current_value
            return 0.0

        error = self.setpoint - current_value
        # Use trapezoidal rule for integral approximation
        new_integral = self.integral + (error + self.prev_error) * dt / 2.0
        derivative = (error - self.prev_error) / dt

        output = (
                self.config.Kp * error +
                self.config.Ki * new_integral +
                self.config.Kd * derivative
        )

        # Apply output limits and anti-windup
        min_limit, max_limit = self.config.output_limits
        if output > max_limit:
            output = max_limit
            # Basic anti-windup: Prevent integral windup when saturated
            # if error * (output - max_limit) < 0: pass # More sophisticated check
            # else: self.integral = new_integral
        elif output < min_limit:
            output = min_limit
            # if error * (output - min_limit) < 0: pass # More sophisticated check
            # else: self.integral = new_integral
        else:
             self.integral = new_integral # Update integral only when not saturated

        self.prev_error = error
        return output


# ========================
# 模糊控制器类 (转速控制)
# ========================
class FuzzyController:
    def __init__(self, setpoint, output_limits):
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.prev_error = 0.0
        self.last_time = None
        self.prev_dRPM_adj = 0.0 # Store previous adjustment for filtering

        # Define universe of discourse and membership functions (simplified triangular/trapezoidal)
        # Error (e): H2S concentration error (setpoint - current)
        # Adjusted universe based on new setpoint (20 ppm) and expected range
        self.e_universe = (-100, 100) # Example range, tune based on expected error around 20 ppm
        self.e_mf = {
            'NB': lambda e: self._trimf(e, -100, -100, -30),
            'NS': lambda e: self._trimf(e, -60, -30, 0),
            'Z':  lambda e: self._trimf(e, -10, 0, 10),
            'PS': lambda e: self._trimf(e, 0, 30, 60),
            'PB': lambda e: self._trimf(e, 30, 100, 100)
        }

        # Change in Error (de): current error - previous error
        self.de_universe = (-30, 30) # Example range, tune based on expected change
        self.de_mf = {
            'NB': lambda de: self._trimf(de, -30, -30, -10),
            'NS': lambda de: self._trimf(de, -20, -10, 0),
            'Z':  lambda de: self._trimf(de, -3, 0, 3),
            'PS': lambda de: self._trimf(de, 0, 10, 20),
            'PB': lambda de: self._trimf(de, 10, 30, 30)
        }

        # Output (dRPM_adj): Change in RPM adjustment
        self.dRPM_adj_universe = (-20, 20) # Example range, tune based on desired adjustment
        self.dRPM_adj_mf = {
            'NB': -15, 'NS': -5, 'Z': 0, 'PS': 5, 'PB': 15
        }

        # Fuzzy Rules (Mamdani-like, simplified to singletons for output)
        # IF e is <e_term> AND de is <de_term> THEN dRPM_adj is <output_term>
        # Adjusted rules based on new target and expected behavior
        self.rules = {
            ('NB', 'NB'): 'PB', ('NB', 'NS'): 'PB', ('NB', 'Z'): 'PS', ('NB', 'PS'): 'Z',  ('NB', 'PB'): 'NS',
            ('NS', 'NB'): 'PB', ('NS', 'NS'): 'PS', ('NS', 'Z'): 'Z',  ('NS', 'PS'): 'NS',  ('NS', 'PB'): 'NB',
            ('Z',  'NB'): 'PS', ('Z', 'NS'): 'Z',  ('Z', 'Z'): 'Z',  ('Z', 'PS'): 'Z',  ('Z', 'PB'): 'NS',
            ('PS', 'NB'): 'NS', ('PS', 'NS'): 'Z',  ('PS', 'Z'): 'NS', ('PS', 'PS'): 'NB',  ('PS', 'PB'): 'NB',
            ('PB', 'NB'): 'Z',  ('PB', 'NS'): 'NS',  ('PB', 'Z'): 'NB', ('PB', 'PS'): 'NB',  ('PB', 'PB'): 'NB'
        }

    def _trimf(self, x, a, b, c):
        """Triangular membership function."""
        return max(0, min((x - a) / (b - a) if b != a else 1, (c - x) / (c - b) if c != b else 1))

    def _fuzzify(self, value, mf_dict):
        """Fuzzify input value using membership functions."""
        return {term: mf(value) for term, mf in mf_dict.items()}

    def _inference(self, fuzzified_e, fuzzified_de):
        """Apply fuzzy rules and get rule strengths."""
        rule_strengths = {}
        for (e_term, de_term), output_term in self.rules.items():
            # Using min for AND operation
            strength = min(fuzzified_e.get(e_term, 0), fuzzified_de.get(de_term, 0))
            if output_term not in rule_strengths or strength > rule_strengths[output_term]:
                 rule_strengths[output_term] = strength
        return rule_strengths

    def _defuzzify(self, rule_strengths):
        """Defuzzify using weighted average (for singleton outputs)."""
        numerator = sum(strength * self.dRPM_adj_mf[term] for term, strength in rule_strengths.items())
        denominator = sum(rule_strengths.values())
        return numerator / denominator if denominator != 0 else 0

    def compute(self, current_value, current_time):
        dt = current_time - self.last_time if self.last_time is not None else 0
        self.last_time = current_time

        error = self.setpoint - current_value
        de = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        # Fuzzification
        fuzzified_e = self._fuzzify(error, self.e_mf)
        fuzzified_de = self._fuzzify(de, self.de_mf)

        # Inference
        rule_strengths = self._inference(fuzzified_e, fuzzified_de)

        # Defuzzification
        dRPM_adj = self._defuzzify(rule_strengths)

        # Apply output limits (optional for d_output, but good practice)
        min_limit, max_limit = self.output_limits
        dRPM_adj = max(min_limit, min(dRPM_adj, max_limit))

        # Simple filtering on the output adjustment
        filtered_dRPM_adj = 0.3 * dRPM_adj + 0.7 * self.prev_dRPM_adj
        self.prev_dRPM_adj = filtered_dRPM_adj

        return filtered_dRPM_adj


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
        # Ensure NTU is not negative or excessively large
        NTU = max(0.0, min(NTU, 20.0)) # Increased cap to potentially allow lower concentrations

        # Mass transfer calculation - ensure stable calculation
        # Avoid division by zero or near zero
        denominator = (1 + NTU / (5 * enhancement_factor))
        if denominator == 0:
            y_out = y_in # No mass transfer if denominator is zero
        else:
            y_out = y_in * math.exp(-NTU / denominator)

        # Removed the final adjustment factor * 0.1 to allow lower concentrations
        outlet_ppm = y_out * 1e6/40 #正确计算

        return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm * 1.2))
    except Exception as e:
        # print(f"Error in calculate_outlet_h2s: {e}")
        return 进口H2S浓度_ppm # Return initial concentration on error


# ========================
# 控制模拟主程序
# ========================
class ControlSystem:
    def __init__(self):
        # Initial process parameters
        self.initial_process_params = {
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
        self.process_params = self.initial_process_params.copy()

        # Single target H2S concentration for all controllers
        self.target_h2s_concentration = 20.0 # Set target to 20 ppm

        # PID controllers for L and G (remain PID for both simulations)
        # Pass the single target concentration as setpoint
        self.pid_L = PIDController(PIDConfig(
            Kp=-0.25, Ki=-0.001, Kd=-0.015,
            output_limits=(-1.5, 1.5)
        ), setpoint=self.target_h2s_concentration)

        self.pid_G = PIDController(PIDConfig(
            Kp=0.01, Ki=0.0003, Kd=0.002,
            output_limits=(-3, 3)
        ), setpoint=self.target_h2s_concentration)

        # PID controller for RPM (used in PID simulation)
        # Pass the single target concentration as setpoint
        self.pid_RPM = PIDController(PIDConfig(
            Kp=-0.8, Ki=-0.03, Kd=-0.6, # Optimized PID parameters for RPM
            output_limits=(-20, 20) # Max RPM adjustment per step
        ), setpoint=self.target_h2s_concentration)

        # Fuzzy controller for RPM (used in Fuzzy simulation)
        # Pass the single target concentration as setpoint
        self.fuzzy_RPM = FuzzyController(
            setpoint=self.target_h2s_concentration, # Target H2S concentration
            output_limits=(-20, 20) # Max RPM adjustment per step
        )

        # Histories for both simulations
        self.history_pid = self._init_history()
        self.history_fuzzy = self._init_history()

        self.prev_adj_pid = {'L': 0, 'RPM': 0, 'G': 0}
        self.prev_adj_fuzzy = {'L': 0, 'RPM': 0, 'G': 0}

    def _init_history(self):
        return {
            '时间': [], '浓度': [],
            '脱硫液流量': [], '转速': [], '煤气流量': []
        }

    def run_simulation(self, duration=600, dt=5, control_type='PID'):
        """Runs a simulation with specified control type for RPM."""
        print(f"Running simulation with {control_type} control for RPM...")
        # Reset parameters and history for the current simulation run
        self.process_params = self.initial_process_params.copy()
        if control_type == 'PID':
            self.history_pid = self._init_history()
            # Re-initialize controllers with the correct setpoint
            self.pid_L = PIDController(PIDConfig(Kp=-0.25, Ki=-0.001, Kd=-0.015, output_limits=(-1.5, 1.5)), setpoint=self.target_h2s_concentration)
            self.pid_G = PIDController(PIDConfig(Kp=0.01, Ki=0.0003, Kd=0.002, output_limits=(-3, 3)), setpoint=self.target_h2s_concentration)
            self.pid_RPM = PIDController(PIDConfig(Kp=-0.8, Ki=-0.03, Kd=-0.6, output_limits=(-20, 20)), setpoint=self.target_h2s_concentration)
            self.prev_adj_pid = {'L': 0, 'RPM': 0, 'G': 0}
        elif control_type == 'Fuzzy':
            self.history_fuzzy = self._init_history()
            # Re-initialize controllers with the correct setpoint
            self.pid_L = PIDController(PIDConfig(Kp=-0.25, Ki=-0.001, Kd=-0.015, output_limits=(-1.5, 1.5)), setpoint=self.target_h2s_concentration)
            self.pid_G = PIDController(PIDConfig(Kp=0.01, Ki=0.0003, Kd=0.002, output_limits=(-3, 3)), setpoint=self.target_h2s_concentration)
            self.fuzzy_RPM = FuzzyController(setpoint=self.target_h2s_concentration, output_limits=(-20, 20))
            self.prev_adj_fuzzy = {'L': 0, 'RPM': 0, 'G': 0}
        else:
            raise ValueError("control_type must be 'PID' or 'Fuzzy'")


        for t in range(0, duration + dt, dt):
            current_conc = calculate_outlet_h2s(**self.process_params)
            self._record_data(t, current_conc, control_type)
            self._adjust_parameters(t, current_conc, control_type)

            # Simulate disturbance: Increase inlet H2S concentration
            if 200 < t < 400:
                 self.process_params["进口H2S浓度_ppm"] = 1100
            else:
                 self.process_params["进口H2S浓度_ppm"] = 1000


    def _record_data(self, t, conc, control_type):
        history = self.history_pid if control_type == 'PID' else self.history_fuzzy
        history['时间'].append(t)
        history['浓度'].append(conc)
        history['脱硫液流量'].append(self.process_params["脱硫液流量_m3h"])
        history['转速'].append(self.process_params["转速_RPM"])
        history['煤气流量'].append(self.process_params["煤气进口流量_m3h"])

    def _adjust_parameters(self, t, current_conc, control_type):
        prev_adj = self.prev_adj_pid if control_type == 'PID' else self.prev_adj_fuzzy

        # PID for Liquid Flow (L) - used in both simulations
        adj_L = self.pid_L.compute(current_conc, t)
        adj_L = 0.2 * adj_L + 0.8 * prev_adj['L']

        # PID for Gas Flow (G) - used in both simulations
        adj_G = self.pid_G.compute(current_conc, t)
        adj_G = 0.2 * adj_G + 0.8 * prev_adj['G']

        # RPM control based on control_type
        if control_type == 'PID':
            adj_RPM = self.pid_RPM.compute(current_conc, t)
            adj_RPM = 0.2 * adj_RPM + 0.8 * prev_adj['RPM']
        elif control_type == 'Fuzzy':
            adj_RPM = self.fuzzy_RPM.compute(current_conc, t)
            # Fuzzy controller already includes internal filtering, but external is fine too
            adj_RPM = 0.2 * adj_RPM + 0.8 * prev_adj['RPM']


        # Apply adjustments with constraints
        self.process_params["脱硫液流量_m3h"] = max(15, min(
            self.process_params["脱硫液流量_m3h"] + adj_L, 40))
        self.process_params["转速_RPM"] = max(1100, min(
            self.process_params["转速_RPM"] + adj_RPM, 1400))
        self.process_params["煤气进口流量_m3h"] = max(700, min(
            self.process_params["煤气进口流量_m3h"] + adj_G, 1000))

        if control_type == 'PID':
            self.prev_adj_pid = {'L': adj_L, 'RPM': adj_RPM, 'G': adj_G}
        elif control_type == 'Fuzzy':
            self.prev_adj_fuzzy = {'L': adj_L, 'RPM': adj_RPM, 'G': adj_G}


    def visualize(self):
        # Configure global style
        plt.style.use('ggplot')
        colors = ['#2c3e50', '#e74c3c', '#27ae60', '#2980b9', '#8e44ad'] # Added a color for comparison line
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'figure.figsize': (12, 8), # Slightly larger figure
            'figure.dpi': 100
        })

        # Plot H2S Concentration Comparison
        plt.figure(1)
        plt.plot(self.history_pid['时间'], self.history_pid['浓度'],
                 color=colors[0], linewidth=2, alpha=0.9, label='PID控制转速')
        plt.plot(self.history_fuzzy['时间'], self.history_fuzzy['浓度'],
                 color=colors[1], linewidth=2, alpha=0.9, label='模糊控制转速')
        plt.axhline(self.target_h2s_concentration, color=colors[2], linestyle='--', linewidth=1.5, label='设定值') # Use a different color for setpoint
        plt.title('H₂S浓度控制效果对比 (PID vs 模糊控制转速)', pad=12)
        plt.xlabel('时间 (秒)')
        plt.ylabel('浓度 (ppm)')
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()

        # Plot RPM Comparison
        plt.figure(2)
        plt.plot(self.history_pid['时间'], self.history_pid['转速'],
                 color=colors[0], linewidth=2, alpha=0.8, label='PID控制转速')
        plt.plot(self.history_fuzzy['时间'], self.history_fuzzy['转速'],
                 color=colors[1], linewidth=2, alpha=0.8, label='模糊控制转速')
        plt.title('转速控制效果对比 (PID vs 模糊)', pad=12)
        plt.xlabel('时间 (秒)')
        plt.ylabel('转速 (RPM)')
        plt.ylim(1100, 1400) # Keep consistent y-limits
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()

        # Plot Liquid Flow (Should be similar for both simulations)
        plt.figure(3)
        plt.plot(self.history_pid['时间'], self.history_pid['脱硫液流量'],
                 color=colors[3], linewidth=2, alpha=0.8, label='PID控制转速')
        plt.plot(self.history_fuzzy['时间'], self.history_fuzzy['脱硫液流量'],
                 color=colors[4], linewidth=2, alpha=0.8, label='模糊控制转速')
        plt.title('脱硫液流量控制', pad=12)
        plt.xlabel('时间 (秒)')
        plt.ylabel('流量 (m³/h)')
        plt.ylim(15, 40) # Keep consistent y-limits
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()

        # Plot Gas Flow (Should be similar for both simulations)
        plt.figure(4)
        plt.plot(self.history_pid['时间'], self.history_pid['煤气流量'],
                 color=colors[3], linewidth=2, alpha=0.8, label='PID控制转速')
        plt.plot(self.history_fuzzy['时间'], self.history_fuzzy['煤气流量'],
                 color=colors[4], linewidth=2, alpha=0.8, label='模糊控制转速')
        plt.title('煤气流量控制', pad=12)
        plt.xlabel('时间 (秒)')
        plt.ylabel('流量 (m³/h)')
        plt.ylim(700, 1000) # Keep consistent y-limits
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()


        plt.show()


# ========================
# 执行程序
# ========================
if __name__ == "__main__":
    # 运行控制模拟
    system = ControlSystem()
    system.run_simulation(duration=600, control_type='PID')
    system.run_simulation(duration=600, control_type='Fuzzy')
    system.visualize()
