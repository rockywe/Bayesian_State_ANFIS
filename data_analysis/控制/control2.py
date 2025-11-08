import math
from dataclasses import dataclass
import time # Needed for PID/Fuzzy time calculation
import copy # Needed for deep copying state in simulation

# ========================
# 核心工艺计算模型 (Core Process Calculation Model)
# ========================
def calculate_outlet_h2s(
        煤气进口流量_m3h,  # m³/h (Gas Inlet Flow)
        进口煤气温度_C,  # ℃ (Inlet Gas Temperature)
        进口煤气压力_kPa,  # kPa (Inlet Gas Pressure)
        脱硫液流量_m3h,  # m³/h (Desulfurization Liquid Flow)
        脱硫液温度_C,  # ℃ (Liquid Temperature)
        脱硫液压力_kPa,  # kPa (Liquid Pressure)
        转速_RPM,  # RPM (Rotational Speed)
        进口H2S浓度_ppm,  # ppm (Inlet H2S Concentration)
        L_exponent=0.6,  # 脱硫液流量指数 (Liquid Flow Exponent)
        RPM_exponent=0.8,  # 转速影响指数 (RPM Influence Exponent)
        G_exponent=-0.25,  # 煤气流量指数 (Gas Flow Exponent)
        gas_velocity_factor=1.2,  # 气体流速因子 (Gas Velocity Factor)
        enhancement_factor=2.5,  # 强化因子 (Enhancement Factor)
        contact_time_base=0.8  # 接触时间基准 (Contact Time Base)
):
    """
    Calculates the estimated H2S outlet concentration based on input parameters.
    根据输入参数计算估算的H2S出口浓度。
    """
    # 物理常数 (Physical Constants)
    D_H2S = 1.8e-9
    H_H2S = 483.0
    alpha = 800
    R_gas = 0.0821
    liquid_density = 1100

    # 设备参数 (Equipment Parameters)
    R_inner = 0.015
    R_outer = 0.85
    h_packing = 0.033
    N_stages = 80

    try:
        # --- Input Validation ---
        # Ensure essential inputs are non-negative
        if any(v < 0 for v in [煤气进口流量_m3h, 进口煤气压力_kPa, 脱硫液流量_m3h, 脱硫液压力_kPa, 转速_RPM, 进口H2S浓度_ppm]):
             print("警告：输入参数包含负值，计算可能不准确。(Warning: Input parameters contain negative values, calculation might be inaccurate.)")
             # Decide handling: return error, default, or clamp to 0? Let's clamp positive inputs.
             煤气进口流量_m3h = max(0, 煤气进口流量_m3h)
             进口煤气压力_kPa = max(0, 进口煤气压力_kPa)
             脱硫液流量_m3h = max(0, 脱硫液流量_m3h)
             脱硫液压力_kPa = max(0, 脱硫液压力_kPa)
             转速_RPM = max(0, 转速_RPM)
             进口H2S浓度_ppm = max(0, 进口H2S浓度_ppm)

        # --- Unit Conversions ---
        G_m3s = 煤气进口流量_m3h / 3600
        T_gas = 进口煤气温度_C + 273.15
        P_total = 进口煤气压力_kPa / 101.325 if 进口煤气压力_kPa > 0 else 0 # Avoid issues with 0 pressure
        y_in = 进口H2S浓度_ppm * 1e-6

        L_m3s = 脱硫液流量_m3h / 3600
        T_liquid = 脱硫液温度_C + 273.15

        R_avg = math.sqrt(R_inner * R_outer)
        omega = 转速_RPM * 2 * math.pi / 60

        # --- Enhanced Mass Transfer Model ---
        # Avoid potential division by zero or invalid operations if flows/RPM are zero
        if G_m3s <= 0 or L_m3s <= 0 or 转速_RPM <= 0:
             # If key inputs are zero, mass transfer is likely negligible or model invalid
             # print("Warning: Zero flow or RPM detected. Assuming outlet = inlet.")
             return float(进口H2S浓度_ppm) # Return inlet concentration

        centrifugal_g = (omega ** 2 * R_avg) / 9.81
        if centrifugal_g <= 0: # Should not happen if RPM > 0, but as safety
             return float(进口H2S浓度_ppm)

        # Use safe power calculation (avoid issues with non-positive bases)
        term_g = centrifugal_g ** (RPM_exponent * enhancement_factor)
        term_l = L_m3s ** (L_exponent * enhancement_factor)
        term_gas = G_m3s ** (G_exponent * enhancement_factor) # G_exponent is negative

        kLa = 0.024 * enhancement_factor * term_g * term_l * term_gas
        kL = kLa / alpha

        # --- Dynamic Material Balance ---
        cross_area = math.pi * (R_outer ** 2 - R_inner ** 2)
        if cross_area <= 0:
            # print("Warning: Invalid cross-sectional area.")
            return float(进口H2S浓度_ppm)

        liquid_velocity = L_m3s / cross_area
        gas_velocity = G_m3s / cross_area

        combined_velocity = (liquid_velocity +
                             gas_velocity_factor * gas_velocity * enhancement_factor)
        if combined_velocity <= 0:
             # print("Warning: Zero or negative combined velocity.")
             return float(进口H2S浓度_ppm) # Or handle differently

        residence_time = contact_time_base * h_packing / combined_velocity

        NTU = kL * alpha * residence_time
        # Ensure NTU is non-negative and reasonably bounded
        NTU = max(0.0, min(NTU, 50.0)) # Increased upper bound slightly

        # --- Mass Transfer Calculation ---
        denominator = (1 + NTU / (5 * enhancement_factor))
        if denominator <= 0: # Avoid division by zero or negative
            # print("Warning: Non-positive denominator in mass transfer calculation.")
            # Decide fallback: maybe exp(-NTU)? or return inlet?
            y_out = y_in * math.exp(-NTU) # Alternative if denominator fails
        else:
            y_out = y_in * math.exp(-NTU / denominator)

        # Using the user's provided calculation with /40 - treat this as empirical calibration
        outlet_ppm = y_out * 1e6 / 40

        # Ensure output is non-negative and not excessively high (e.g., capped by inlet)
        return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm * 1.2)) # Allow slightly higher for model quirks

    except ValueError as ve:
        print(f"计算中出现数值错误 (Numerical error during calculation): {ve}")
        return float(进口H2S浓度_ppm) # Fallback on math errors
    except Exception as e:
        print(f"计算 H2S 出口浓度时发生意外错误 (Unexpected error calculating H2S outlet concentration): {e}")
        # Return initial concentration or a specific error indicator if preferred
        return float(进口H2S浓度_ppm) # General fallback

# ========================
# PID控制器类 (PID Controller Class)
# ========================
@dataclass
class PIDConfig:
    Kp: float
    Ki: float
    Kd: float
    output_limits: tuple # Min and Max adjustment per step

class PIDController:
    def __init__(self, config: PIDConfig, setpoint: float):
        self.config = config
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

    def compute(self, current_value, current_time):
        """Calculates the PID adjustment."""
        # Handle time calculation robustly
        if self.last_time is None:
            dt = 1.0 # Assume 1 second interval for the first call
        else:
            dt = current_time - self.last_time
            if dt <= 0: # If time hasn't advanced or went backward, skip derivative/integral update
                dt = 1.0 # Prevent division by zero, but don't update integral/derivative meaningfully

        self.last_time = current_time # Store current time for next calculation

        error = self.setpoint - current_value

        # --- Integral Calculation ---
        # Use trapezoidal rule for integration
        # Only integrate if dt is positive and reasonable
        if dt > 0 and dt < 100: # Avoid huge integral steps if time jumps
            new_integral = self.integral + (error + self.prev_error) * dt / 2.0
        else:
            new_integral = self.integral # Keep integral as is if dt is invalid

        # --- Derivative Calculation ---
        if dt > 0 and dt < 100: # Calculate derivative only if dt is valid
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0 # No derivative if time step is invalid

        # --- PID Output Calculation ---
        output = (
                self.config.Kp * error +
                self.config.Ki * new_integral +
                self.config.Kd * derivative
        )

        # --- Output Limiting and Anti-windup ---
        min_limit, max_limit = self.config.output_limits
        if output > max_limit:
            output = max_limit
            # Anti-windup: Only allow integral to decrease if error and output have opposite signs
            if error * output > 0: # If error contributes to saturation, freeze integral
                 new_integral = self.integral
        elif output < min_limit:
            output = min_limit
            # Anti-windup
            if error * output > 0: # If error contributes to saturation, freeze integral
                 new_integral = self.integral

        # Update integral state only if it wasn't frozen by anti-windup
        self.integral = new_integral

        # --- Store previous error for next iteration ---
        self.prev_error = error

        return output

    def reset(self):
        """Resets the controller's internal state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

# ========================
# 模糊控制器类 (转速控制) (Fuzzy Controller Class for RPM Control)
# ========================
class FuzzyController:
    def __init__(self, setpoint, output_limits):
        self.setpoint = setpoint
        self.output_limits = output_limits # Min/Max adjustment per step
        self.prev_error = 0.0
        self.last_time = None
        self.prev_dRPM_adj = 0.0 # For filtering output

        # Define universe of discourse and membership functions (simplified triangular/trapezoidal)
        # Error (e): H2S concentration error (setpoint - current)
        self.e_universe = (-100, 100) # Expected range of error
        self.e_mf = {
            'NB': lambda e: self._trimf(e, -100, -100, -30), # Negative Big
            'NS': lambda e: self._trimf(e, -60, -30, 0),    # Negative Small
            'Z':  lambda e: self._trimf(e, -10, 0, 10),     # Zero
            'PS': lambda e: self._trimf(e, 0, 30, 60),      # Positive Small
            'PB': lambda e: self._trimf(e, 30, 100, 100)     # Positive Big
        }

        # Change in Error (de): current error - previous error
        self.de_universe = (-30, 30) # Expected range of change in error
        self.de_mf = {
            'NB': lambda de: self._trimf(de, -30, -30, -10),
            'NS': lambda de: self._trimf(de, -20, -10, 0),
            'Z':  lambda de: self._trimf(de, -3, 0, 3),
            'PS': lambda de: self._trimf(de, 0, 10, 20),
            'PB': lambda de: self._trimf(de, 10, 30, 30)
        }

        # Output (dRPM_adj): Change in RPM adjustment (Singleton values for defuzzification)
        self.dRPM_adj_universe = (-100, 100) # Range of adjustment output
        self.dRPM_adj_mf = {
            'NB': -90, 'NS': -40, 'Z': 0, 'PS': 40, 'PB': 90
        }

        # Fuzzy Rules (Mamdani-like, simplified to singletons for output)
        # Format: (Error Term, Change in Error Term) -> Output Term
        self.rules = {
            # If Error is NB (very low concentration) -> Increase RPM significantly
            ('NB', 'NB'): 'PB', ('NB', 'NS'): 'PB', ('NB', 'Z'): 'PB', ('NB', 'PS'): 'PS', ('NB', 'PB'): 'Z',
            # If Error is NS (low concentration) -> Increase RPM moderately
            ('NS', 'NB'): 'PB', ('NS', 'NS'): 'PS', ('NS', 'Z'): 'PS', ('NS', 'PS'): 'Z', ('NS', 'PB'): 'NS',
            # If Error is Z (around setpoint) -> Small adjustments or zero
            ('Z',  'NB'): 'PS', ('Z', 'NS'): 'Z',  ('Z', 'Z'): 'Z',  ('Z', 'PS'): 'Z', ('Z', 'PB'): 'NS',
            # If Error is PS (high concentration) -> Decrease RPM moderately
            ('PS', 'NB'): 'NS', ('PS', 'NS'): 'Z',  ('PS', 'Z'): 'NS', ('PS', 'PS'): 'NB', ('PS', 'PB'): 'NB',
            # If Error is PB (very high concentration) -> Decrease RPM significantly
            ('PB', 'NB'): 'Z',  ('PB', 'NS'): 'NB', ('PB', 'Z'): 'NB', ('PB', 'PS'): 'NB', ('PB', 'PB'): 'NB'
        }

    def _trimf(self, x, a, b, c):
        """Triangular membership function."""
        try:
            # Ensure a <= b <= c
            a, b, c = sorted((a, b, c))
            if a == c: # Handle case of a single point (delta function approximation)
                return 1.0 if x == a else 0.0

            val1 = (x - a) / (b - a) if b != a else (1.0 if x >= b else 0.0)
            val2 = (c - x) / (c - b) if c != b else (1.0 if x <= b else 0.0)
            # Ensure results are within [0, 1] and handle vertical slopes
            return max(0.0, min(val1 if b != a else (1.0 if x >= b else 0.0),
                                val2 if c != b else (1.0 if x <= b else 0.0)))
        except ZeroDivisionError:
             # This case should be covered by a==c or the ternary operators, but added as safety
             return 1.0 if a <= x <= c else 0.0

    def _fuzzify(self, value, mf_dict, universe):
        """Fuzzify input value using membership functions."""
        min_u, max_u = universe
        clamped_value = max(min_u, min(value, max_u))
        return {term: mf(clamped_value) for term, mf in mf_dict.items()}

    def _inference(self, fuzzified_e, fuzzified_de):
        """Apply fuzzy rules and get rule strengths (using MIN for AND)."""
        rule_strengths = {} # Stores strength for each *output* term
        for (e_term, de_term), output_term in self.rules.items():
            e_strength = fuzzified_e.get(e_term, 0)
            de_strength = fuzzified_de.get(de_term, 0)
            strength = min(e_strength, de_strength)
            if strength > 0:
                # Aggregate strength for the output term (MAX operator)
                if output_term not in rule_strengths or strength > rule_strengths[output_term]:
                     rule_strengths[output_term] = strength
        return rule_strengths

    def _defuzzify(self, rule_strengths):
        """Defuzzify using weighted average (Center of Gravity for Singletons)."""
        numerator = 0
        denominator = 0
        if not rule_strengths:
            return 0.0
        for term, strength in rule_strengths.items():
            if term in self.dRPM_adj_mf:
                numerator += strength * self.dRPM_adj_mf[term]
                denominator += strength
            else:
                print(f"警告：在输出隶属度函数中未找到项 '{term}' (Warning: Term '{term}' not found in output membership functions)")
        return numerator / denominator if denominator != 0 else 0

    def compute(self, current_value, current_time):
        """Computes the fuzzy logic adjustment for RPM."""
        if self.last_time is None:
            dt = 1.0 # Assume 1 second interval for the first call
        else:
            dt = current_time - self.last_time
            if dt <= 0:
                 dt = 1.0 # Prevent division by zero or invalid dError

        self.last_time = current_time

        error = self.setpoint - current_value
        de = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        # --- Fuzzification ---
        fuzzified_e = self._fuzzify(error, self.e_mf, self.e_universe)
        fuzzified_de = self._fuzzify(de, self.de_mf, self.de_universe)

        # --- Inference ---
        rule_strengths = self._inference(fuzzified_e, fuzzified_de)

        # --- Defuzzification ---
        dRPM_adj = self._defuzzify(rule_strengths)

        # --- Output Limiting ---
        min_limit, max_limit = self.output_limits
        dRPM_adj = max(min_limit, min(dRPM_adj, max_limit))

        # --- Output Filtering (Simple Moving Average like original) ---
        # Use filtering similar to the original simulation code for smoother output
        # Note: Original fuzzy simulation used 0.9*current + 0.1*previous
        filtered_dRPM_adj = 0.9 * dRPM_adj + 0.1 * self.prev_dRPM_adj
        self.prev_dRPM_adj = filtered_dRPM_adj # Store filtered value for next step's filter

        # Return the raw adjustment for "next step", filtering applied in simulation/final calc
        # Return filtered adjustment directly as per original simulation's fuzzy step
        return filtered_dRPM_adj

    def reset(self):
        """Resets the controller's internal state."""
        self.prev_error = 0.0
        self.last_time = None
        self.prev_dRPM_adj = 0.0

# ========================
# 模拟运行函数 (Simulation Function)
# ========================
def run_prediction_simulation(initial_params, target_h2s, rpm_control_type, duration_s=600, dt_s=5):
    """
    Runs a simulation loop to predict future state.
    运行模拟循环以预测未来状态。

    Args:
        initial_params (dict): Dictionary of starting process parameters.
        target_h2s (float): The target H2S concentration (ppm).
        rpm_control_type (str): 'pid' or 'fuzzy'.
        duration_s (int): Simulation duration in seconds.
        dt_s (int): Time step in seconds.

    Returns:
        dict: Final process parameters after the simulation.
        list: History of H2S concentration during simulation.
    """
    print(f"\n--- 运行 {rpm_control_type.upper()} 全程预测 ({duration_s} 秒) ---")
    print(f"--- Running {rpm_control_type.upper()} Full Prediction ({duration_s}s) ---")

    process_params = copy.deepcopy(initial_params) # Work on a copy
    h2s_history = []

    # Initialize controllers
    # Liquid flow is always PID controlled in this setup
    pid_L_config = PIDConfig(Kp=-0.25, Ki=-0.001, Kd=-0.015, output_limits=(-1.5, 1.5))
    pid_L = PIDController(pid_L_config, setpoint=target_h2s)

    # RPM controller based on selection
    if rpm_control_type == 'pid':
        pid_rpm_config = PIDConfig(Kp=-0.8, Ki=-0.03, Kd=-0.6, output_limits=(-30, 30))
        rpm_controller = PIDController(pid_rpm_config, setpoint=target_h2s)
    elif rpm_control_type == 'fuzzy':
        rpm_controller = FuzzyController(setpoint=target_h2s, output_limits=(-50, 50))
    else:
        raise ValueError("Invalid rpm_control_type. Must be 'pid' or 'fuzzy'.")

    # Simulation loop
    current_time_sim = 0 # Relative time for simulation steps
    pid_L.reset()
    rpm_controller.reset()

    # Store initial state
    initial_conc = calculate_outlet_h2s(**process_params)
    h2s_history.append({'time': 0, 'conc': initial_conc})
    print(f"时间 0s: H2S = {initial_conc:.2f} ppm, 转速 = {process_params['转速_RPM']:.1f} RPM, 流量 = {process_params['脱硫液流量_m3h']:.1f} m³/h")

    for t_step in range(dt_s, duration_s + dt_s, dt_s):
        current_time_sim += dt_s
        # 1. Calculate current concentration based on last step's parameters
        current_conc = calculate_outlet_h2s(**process_params)
        h2s_history.append({'time': t_step, 'conc': current_conc})

        # 2. Compute adjustments using controllers
        adj_L = pid_L.compute(current_conc, current_time_sim)
        adj_RPM = rpm_controller.compute(current_conc, current_time_sim) # Fuzzy compute already includes filtering

        # 3. Apply adjustments and constraints to get parameters for *next* step
        # Apply filtering for PID RPM adjustment (like original simulation)
        # Note: Fuzzy controller's compute method already applies filtering internally
        # We need a way to store previous adjustment if using PID
        # Let's simplify: apply raw adjustment for PID here too for prediction consistency
        # Or re-introduce prev_adj tracking if needed for accuracy matching original sim

        process_params["脱硫液流量_m3h"] = max(0.0, min(
            process_params["脱硫液流量_m3h"] + adj_L, 60.0)) # Flow range 0-60
        process_params["转速_RPM"] = max(0.0, min(
            process_params["转速_RPM"] + adj_RPM, 900.0)) # RPM range 0-900

        # Optional: Print progress every minute
        if t_step % 60 == 0:
             print(f"时间 {t_step}s: H2S = {current_conc:.2f} ppm, 转速 = {process_params['转速_RPM']:.1f} RPM, 流量 = {process_params['脱硫液流量_m3h']:.1f} m³/h")


    final_conc = calculate_outlet_h2s(**process_params) # Calculate concentration with final parameters
    print(f"预测结束 (Prediction End): 时间 {duration_s}s")
    print(f"  - 最终预测 H2S 浓度 (Final Predicted H2S): {final_conc:.2f} ppm")
    print(f"  - 最终预测 转速 (Final Predicted RPM): {process_params['转速_RPM']:.2f} RPM")
    print(f"  - 最终预测 脱硫液流量 (Final Predicted Liquid Flow): {process_params['脱硫液流量_m3h']:.2f} m³/h")
    print("-" * 20)

    return process_params, h2s_history


# ========================
# 用户交互界面函数 (User Interface Functions)
# ========================

def get_float_input(prompt, default=None):
    """Safely gets a float input from the user, with optional default."""
    while True:
        try:
            if default is not None:
                value_str = input(f"{prompt} (默认 Default: {default}): ").strip()
                if not value_str: # User pressed Enter
                    return float(default)
            else:
                 value_str = input(prompt).strip()

            value = float(value_str)
            return value
        except ValueError:
            print("无效输入，请输入一个数字。(Invalid input, please enter a number.)")
        except Exception as e:
             print(f"发生错误 (An error occurred): {e}")


def calculate_h2s_ui():
    """Handles UI for calculating H2S outlet concentration."""
    print("\n--- 计算H2S出口浓度 (Calculate H2S Outlet Concentration) ---")
    params = {}
    # Provide defaults based on typical ranges or last known good values if available
    params['煤气进口流量_m3h'] = get_float_input("请输入 煤气进口流量 (m³/h): ", 1500)
    params['进口煤气温度_C'] = get_float_input("请输入 进口煤气温度 (℃): ", 40)
    params['进口煤气压力_kPa'] = get_float_input("请输入 进口煤气压力 (kPa): ", 20)
    params['脱硫液流量_m3h'] = get_float_input("请输入 脱硫液流量 (m³/h): ", 30)
    params['脱硫液温度_C'] = get_float_input("请输入 脱硫液温度 (℃): ", 35)
    params['脱硫液压力_kPa'] = get_float_input("请输入 脱硫液压力 (kPa): ", 40)
    params['转速_RPM'] = get_float_input("请输入 转速 (RPM): ", 450)
    params['进口H2S浓度_ppm'] = get_float_input("请输入 进口H2S浓度 (ppm): ", 1000)

    outlet_concentration = calculate_outlet_h2s(**params)

    print("-" * 20)
    if outlet_concentration is not None:
        print(f"计算得到的 H2S 出口浓度为 (Calculated H2S Outlet Concentration): {outlet_concentration:.2f} ppm")
    else:
        print("计算出错。(Error during calculation.)")
    print("-" * 20)

def intelligent_control_ui():
    """Handles UI for intelligent control recommendations and predictions."""
    print("\n--- 智能控制 (Intelligent Control) ---")

    # --- Get Control Mode ---
    print("请选择控制模式:")
    print("  1. PID 下一步预测 (PID Next Step Prediction)")
    print("  2. Fuzzy 下一步预测 (Fuzzy Next Step Prediction)")
    print("  3. 最优化 下一步预测 (Optimization Next Step Prediction) (RPM: 700-800)")
    print("  4. PID 全程预测 (PID Full Prediction) (Simulate 10 min)")
    print("  5. Fuzzy 全程预测 (Fuzzy Full Prediction) (Simulate 10 min)")

    while True:
        mode_choice = input("请选择一个模式 (1-5): ").strip()
        if mode_choice in ['1', '2', '3', '4', '5']:
            break
        else:
            print("无效选择，请输入 1 到 5 之间的数字。(Invalid choice, please enter a number between 1 and 5.)")

    # --- Get Common Inputs ---
    target_h2s = get_float_input("请输入 目标H2S浓度 (ppm) Target H2S Concentration: ", 20)
    current_h2s = get_float_input("请输入 当前H2S浓度 (ppm) Current H2S Concentration: ")
    current_rpm = get_float_input("请输入 当前转速 (RPM) Current RPM: ", 450)
    current_flow = get_float_input("请输入 当前脱硫液流量 (m³/h) Current Liquid Flow: ", 30)

    # --- Store current parameters for simulation ---
    # Need other parameters for calculate_outlet_h2s inside simulation
    # Get these with defaults or ask user
    print("\n请输入当前其他工艺参数 (用于模拟) Enter other current process parameters (for simulation):")
    current_process_params = {
        "煤气进口流量_m3h": get_float_input("  - 煤气进口流量 (m³/h) Gas Inlet Flow: ", 1500),
        "进口煤气温度_C": get_float_input("  - 进口煤气温度 (℃) Inlet Gas Temp: ", 40),
        "进口煤气压力_kPa": get_float_input("  - 进口煤气压力 (kPa) Inlet Gas Pressure: ", 20),
        "脱硫液流量_m3h": current_flow, # Use the already entered value
        "脱硫液温度_C": get_float_input("  - 脱硫液温度 (℃) Liquid Temp: ", 35),
        "脱硫液压力_kPa": get_float_input("  - 脱硫液压力 (kPa) Liquid Pressure: ", 40),
        "转速_RPM": current_rpm, # Use the already entered value
        "进口H2S浓度_ppm": get_float_input("  - 进口H2S浓度 (ppm) Inlet H2S Conc: ", 1000),
        # Keep model parameters default unless specified otherwise
        "L_exponent": 0.6,
        "RPM_exponent": 0.8,
        "G_exponent": -0.25,
        "gas_velocity_factor": 1.2,
        "enhancement_factor": 2.5,
        "contact_time_base": 0.8
    }


    # --- Handle Full Prediction Modes ---
    if mode_choice == '4': # PID Full Prediction
        run_prediction_simulation(current_process_params, target_h2s, 'pid', duration_s=600, dt_s=5)
        return # End UI function after simulation output
    elif mode_choice == '5': # Fuzzy Full Prediction
        run_prediction_simulation(current_process_params, target_h2s, 'fuzzy', duration_s=600, dt_s=5)
        return # End UI function after simulation output

    # --- Handle Next Step Prediction Modes (1, 2, 3) ---
    control_type = 'pid' # Default for mode 1 and 3
    if mode_choice == '2':
        control_type = 'fuzzy'

    is_optimization = (mode_choice == '3')

    # --- Initialize Controllers for Next Step ---
    # Liquid flow is always PID controlled
    pid_L_config = PIDConfig(Kp=-0.25, Ki=-0.001, Kd=-0.015, output_limits=(-1.5, 1.5))
    pid_L = PIDController(pid_L_config, setpoint=target_h2s)

    rpm_controller = None
    if control_type == 'pid': # Includes Optimization mode base
        pid_rpm_config = PIDConfig(Kp=-0.8, Ki=-0.03, Kd=-0.6, output_limits=(-30, 30))
        rpm_controller = PIDController(pid_rpm_config, setpoint=target_h2s)
    else: # fuzzy
        rpm_controller = FuzzyController(setpoint=target_h2s, output_limits=(-50, 50))

    # --- Compute Adjustments for Next Step ---
    current_time = time.time() # Use current time

    # Reset controllers for a clean calculation based only on current state
    pid_L.reset()
    rpm_controller.reset()

    # Compute adjustments based on the *current* error
    adj_L = pid_L.compute(current_h2s, current_time)
    adj_RPM = rpm_controller.compute(current_h2s, current_time)

    # --- Calculate Recommended Next Values ---
    recommended_flow = max(0.0, min(current_flow + adj_L, 60.0)) # Flow range 0-60
    recommended_rpm = max(0.0, min(current_rpm + adj_RPM, 900.0)) # RPM range 0-900

    rpm_clamped = False
    if is_optimization:
        rpm_lower_bound = 700.0
        rpm_upper_bound = 800.0
        if recommended_rpm < rpm_lower_bound:
            recommended_rpm = rpm_lower_bound
            rpm_clamped = True
            print("提示: 推荐转速已调整至下限 700 RPM。(Info: Recommended RPM clamped to lower bound 700 RPM.)")
            # Optional: Could try recalculating adj_L with stronger gains if RPM is clamped
        elif recommended_rpm > rpm_upper_bound:
            recommended_rpm = rpm_upper_bound
            rpm_clamped = True
            print("提示: 推荐转速已调整至上限 800 RPM。(Info: Recommended RPM clamped to upper bound 800 RPM.)")
            # Optional: Could try recalculating adj_L with stronger gains if RPM is clamped

    # --- Output Recommendations for Next Step ---
    print("-" * 20)
    mode_desc = ""
    if mode_choice == '1': mode_desc = "PID Next Step"
    elif mode_choice == '2': mode_desc = "Fuzzy Next Step"
    elif mode_choice == '3': mode_desc = "Optimization Next Step (RPM 700-800)"

    print(f"基于当前状态和目标值 {target_h2s} ppm (模式: {mode_desc})")
    print(f"(Based on current state and target {target_h2s} ppm) (Mode: {mode_desc})")
    print(f"  - 建议的下一步脱硫液流量调整量 (Recommended next liquid flow adjustment): {adj_L:+.2f} m³/h")
    print(f"  - 建议的下一步转速调整量 (Recommended next RPM adjustment): {adj_RPM:+.2f} RPM")
    print("-" * 20)
    print(f"  - 建议的下一步脱硫液流量 (Recommended next liquid flow rate): {recommended_flow:.2f} m³/h")
    print(f"  - 建议的下一步转速 (Recommended next RPM): {recommended_rpm:.2f} RPM {'(已约束 Clamped)' if rpm_clamped else ''}")
    print("-" * 20)


# ========================
# 主程序入口 (Main Program Entry Point)
# ========================
def main():
    """Main function to run the interactive system."""
    while True:
        print("\n========== H2S 控制系统菜单 (H2S Control System Menu) ==========")
        print("1. 计算H2S出口浓度 (Calculate H2S Outlet Concentration)")
        print("2. 智能控制与预测 (Intelligent Control & Prediction)")
        print("3. 退出 (Exit)")
        print("==============================================================")

        choice = input("请选择一个选项 (1/2/3): ").strip()

        if choice == '1':
            calculate_h2s_ui()
        elif choice == '2':
            intelligent_control_ui()
        elif choice == '3':
            print("正在退出程序... (Exiting program...)")
            break
        else:
            print("无效选择，请输入 1, 2, 或 3。(Invalid choice, please enter 1, 2, or 3.)")

        # Add a small delay before prompting to continue, makes UI feel less abrupt
        time.sleep(0.5)
        input("\n按 Enter 键继续... (Press Enter to continue...)") # Pause for user to read output

if __name__ == "__main__":
    main()
