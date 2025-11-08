import math
from dataclasses import dataclass
import time # Needed for PID/Fuzzy time calculation

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
    D_H2S = 1.8e-9  # H2S扩散系数 (m²/s) (H2S Diffusion Coefficient)
    H_H2S = 483.0  # 亨利常数 (atm·m³/mol) (Henry's Constant)
    alpha = 800  # 有效比表面积 (m²/m³) (Effective Specific Surface Area)
    R_gas = 0.0821  # 气体常数 (L·atm/mol/K) (Gas Constant)
    liquid_density = 1100  # 脱硫液密度 (kg/m³) (Liquid Density)

    # 设备参数 (Equipment Parameters)
    R_inner = 0.015  # 转子内径 (m) (Rotor Inner Radius)
    R_outer = 0.85  # 转子外径 (m) (Rotor Outer Radius)
    h_packing = 0.033  # 填料高度 (m) (Packing Height)
    N_stages = 80  # 理论级数 (Theoretical Stages)

    try:
        # 单位转换 (Unit Conversions)
        G_m3s = 煤气进口流量_m3h / 3600
        T_gas = 进口煤气温度_C + 273.15
        P_total = 进口煤气压力_kPa / 101.325
        y_in = 进口H2S浓度_ppm * 1e-6

        L_m3s = 脱硫液流量_m3h / 3600
        T_liquid = 脱硫液温度_C + 273.15

        R_avg = math.sqrt(R_inner * R_outer)
        omega = 转速_RPM * 2 * math.pi / 60

        # 增强传质模型 (Enhanced Mass Transfer Model)
        centrifugal_g = (omega ** 2 * R_avg) / 9.81
        # Avoid potential division by zero or invalid operations if flows/RPM are zero
        if G_m3s <= 0 or L_m3s <= 0 or centrifugal_g <= 0:
             # If inputs lead to invalid intermediate values, return inlet concentration
             # print("Warning: Zero or invalid flow/RPM detected in kLa calculation.")
             return float(进口H2S浓度_ppm)

        kLa = 0.024 * enhancement_factor * (
            centrifugal_g ** (RPM_exponent * enhancement_factor) *
            L_m3s ** (L_exponent * enhancement_factor) *
            G_m3s ** (G_exponent * enhancement_factor)
        )
        kL = kLa / alpha

        # 动态物料平衡 (Dynamic Material Balance)
        cross_area = math.pi * (R_outer ** 2 - R_inner ** 2)
        if cross_area == 0:
            # print("Warning: Zero cross-sectional area.")
            return float(进口H2S浓度_ppm)

        liquid_velocity = L_m3s / cross_area
        gas_velocity = G_m3s / cross_area

        combined_velocity = (liquid_velocity +
                             gas_velocity_factor * gas_velocity * enhancement_factor)
        if combined_velocity == 0:
             # print("Warning: Zero combined velocity.")
             return float(进口H2S浓度_ppm)

        residence_time = contact_time_base * h_packing / combined_velocity

        NTU = kL * alpha * residence_time
        # Ensure NTU is not negative or excessively large
        NTU = max(0.0, min(NTU, 20.0))

        # Mass transfer calculation - ensure stable calculation
        # Simplified calculation based on original code's structure
        # The original exp(-NTU / denominator) with denominator = (1 + NTU / (5 * enhancement_factor))
        # and the final division by 40 seems specific to the original model's calibration.
        # We keep it here as per the reference code.
        denominator = (1 + NTU / (5 * enhancement_factor))
        if denominator == 0:
            y_out = y_in
        else:
            y_out = y_in * math.exp(-NTU / denominator)

        # Using the user's provided calculation with /40
        outlet_ppm = y_out * 1e6 / 40

        # Ensure output is non-negative and not excessively high
        return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm * 1.2))

    except Exception as e:
        print(f"计算 H2S 出口浓度时出错 (Error calculating H2S outlet concentration): {e}")
        # Return initial concentration or a specific error indicator if preferred
        return float(进口H2S浓度_ppm)

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
        dt = current_time - self.last_time if self.last_time is not None else 1.0 # Assume dt=1 for first call
        self.last_time = current_time # Store current time for next calculation

        error = self.setpoint - current_value

        # --- Integral Calculation ---
        # Use trapezoidal rule for integration
        new_integral = self.integral + (error + self.prev_error) * dt / 2.0

        # --- Derivative Calculation ---
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0 # Avoid division by zero

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
            # Anti-windup: Prevent integral from growing excessively if output is saturated
            # Only integrate if output is within limits
        elif output < min_limit:
            output = min_limit
            # Anti-windup
        else:
             # Only update integral if output is not saturated
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
            # If Error is NB (very low concentration)
            ('NB', 'NB'): 'PB', ('NB', 'NS'): 'PB', ('NB', 'Z'): 'PB', ('NB', 'PS'): 'PS', ('NB', 'PB'): 'Z',
            # If Error is NS (low concentration)
            ('NS', 'NB'): 'PB', ('NS', 'NS'): 'PS', ('NS', 'Z'): 'PS', ('NS', 'PS'): 'Z', ('NS', 'PB'): 'NS',
            # If Error is Z (around setpoint)
            ('Z',  'NB'): 'PS', ('Z', 'NS'): 'Z',  ('Z', 'Z'): 'Z',  ('Z', 'PS'): 'Z', ('Z', 'PB'): 'NS',
            # If Error is PS (high concentration)
            ('PS', 'NB'): 'NS', ('PS', 'NS'): 'Z',  ('PS', 'Z'): 'NS', ('PS', 'PS'): 'NB', ('PS', 'PB'): 'NB',
            # If Error is PB (very high concentration)
            ('PB', 'NB'): 'Z',  ('PB', 'NS'): 'NB', ('PB', 'Z'): 'NB', ('PB', 'PS'): 'NB', ('PB', 'PB'): 'NB'
        }

    def _trimf(self, x, a, b, c):
        """Triangular membership function."""
        # Clamps x within the universe before calculating membership
        # x = max(min(x, c), a) # Optional: Clamp input to universe boundaries
        try:
            val1 = (x - a) / (b - a) if b != a else (1.0 if x >= b else 0.0)
            val2 = (c - x) / (c - b) if c != b else (1.0 if x <= b else 0.0)
            return max(0, min(val1, val2))
        except ZeroDivisionError:
            return 1.0 if a <= x <= c else 0.0 # Handle cases where a=b or b=c

    def _fuzzify(self, value, mf_dict, universe):
        """Fuzzify input value using membership functions."""
        # Clamp value to the defined universe
        min_u, max_u = universe
        clamped_value = max(min_u, min(value, max_u))
        return {term: mf(clamped_value) for term, mf in mf_dict.items()}

    def _inference(self, fuzzified_e, fuzzified_de):
        """Apply fuzzy rules and get rule strengths (using MIN for AND)."""
        rule_strengths = {} # Stores strength for each *output* term
        active_rules_count = 0
        for (e_term, de_term), output_term in self.rules.items():
            # Get membership degrees for the rule's antecedents
            e_strength = fuzzified_e.get(e_term, 0)
            de_strength = fuzzified_de.get(de_term, 0)

            # Calculate firing strength of the rule (MIN operator for AND)
            strength = min(e_strength, de_strength)

            if strength > 0:
                active_rules_count += 1
                # Aggregate strength for the output term (MAX operator for OR-ing rules leading to the same output)
                if output_term not in rule_strengths or strength > rule_strengths[output_term]:
                     rule_strengths[output_term] = strength
        # print(f"Active rules: {active_rules_count}") # Debugging
        # print(f"Rule strengths: {rule_strengths}") # Debugging
        return rule_strengths

    def _defuzzify(self, rule_strengths):
        """Defuzzify using weighted average (Center of Gravity for Singletons)."""
        numerator = 0
        denominator = 0
        if not rule_strengths: # Handle case where no rules fired
            # print("Warning: No fuzzy rules fired.")
            return 0.0

        for term, strength in rule_strengths.items():
            # Ensure the term exists in the output MF dictionary
            if term in self.dRPM_adj_mf:
                numerator += strength * self.dRPM_adj_mf[term]
                denominator += strength
            else:
                print(f"警告：在输出隶属度函数中未找到项 '{term}' (Warning: Term '{term}' not found in output membership functions)")


        # Avoid division by zero if total strength is zero (shouldn't happen if rule_strengths is not empty)
        return numerator / denominator if denominator != 0 else 0

    def compute(self, current_value, current_time):
        """Computes the fuzzy logic adjustment for RPM."""
        dt = current_time - self.last_time if self.last_time is not None else 1.0 # Assume dt=1 for first call
        self.last_time = current_time

        error = self.setpoint - current_value
        de = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error # Store current error for next calculation

        # --- Fuzzification ---
        fuzzified_e = self._fuzzify(error, self.e_mf, self.e_universe)
        fuzzified_de = self._fuzzify(de, self.de_mf, self.de_universe)
        # print(f"Time: {current_time}, Error: {error:.2f}, dError: {de:.2f}") # Debugging
        # print(f"Fuzzified E: { {k: round(v, 2) for k, v in fuzzified_e.items()} }") # Debugging
        # print(f"Fuzzified dE: { {k: round(v, 2) for k, v in fuzzified_de.items()} }") # Debugging

        # --- Inference ---
        rule_strengths = self._inference(fuzzified_e, fuzzified_de)

        # --- Defuzzification ---
        dRPM_adj = self._defuzzify(rule_strengths)
        # print(f"Raw dRPM_adj: {dRPM_adj:.2f}") # Debugging

        # --- Output Limiting ---
        min_limit, max_limit = self.output_limits
        dRPM_adj = max(min_limit, min(dRPM_adj, max_limit))

        # --- Output Filtering (Simple Moving Average) ---
        # Use filtering similar to the original simulation code for smoother output
        filtered_dRPM_adj = 0.9 * dRPM_adj + 0.1 * self.prev_dRPM_adj
        self.prev_dRPM_adj = filtered_dRPM_adj # Store for next filtering step
        # print(f"Filtered dRPM_adj: {filtered_dRPM_adj:.2f}") # Debugging

        return filtered_dRPM_adj # Return the filtered adjustment

    def reset(self):
        """Resets the controller's internal state."""
        self.prev_error = 0.0
        self.last_time = None
        self.prev_dRPM_adj = 0.0

# ========================
# 用户交互界面函数 (User Interface Functions)
# ========================

def get_float_input(prompt):
    """Safely gets a float input from the user."""
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("无效输入，请输入一个数字。(Invalid input, please enter a number.)")

def calculate_h2s_ui():
    """Handles UI for calculating H2S outlet concentration."""
    print("\n--- 计算H2S出口浓度 (Calculate H2S Outlet Concentration) ---")
    params = {}
    params['煤气进口流量_m3h'] = get_float_input("请输入 煤气进口流量 (m³/h): ")
    params['进口煤气温度_C'] = get_float_input("请输入 进口煤气温度 (℃): ")
    params['进口煤气压力_kPa'] = get_float_input("请输入 进口煤气压力 (kPa): ")
    params['脱硫液流量_m3h'] = get_float_input("请输入 脱硫液流量 (m³/h): ")
    params['脱硫液温度_C'] = get_float_input("请输入 脱硫液温度 (℃): ")
    params['脱硫液压力_kPa'] = get_float_input("请输入 脱硫液压力 (kPa): ")
    params['转速_RPM'] = get_float_input("请输入 转速 (RPM): ")
    params['进口H2S浓度_ppm'] = get_float_input("请输入 进口H2S浓度 (ppm): ")

    # Optional: Ask for other parameters or use defaults
    # params['L_exponent'] = get_float_input("Enter L_exponent (default 0.6): ") or 0.6
    # ... and so on for other parameters if needed

    outlet_concentration = calculate_outlet_h2s(**params)

    print("-" * 20)
    print(f"计算得到的 H2S 出口浓度为 (Calculated H2S Outlet Concentration): {outlet_concentration:.2f} ppm")
    print("-" * 20)

def intelligent_control_ui():
    """Handles UI for intelligent control recommendations."""
    print("\n--- 智能控制 (Intelligent Control) ---")

    # --- Get Control Type ---
    while True:
        control_choice = input("请选择转速控制方式 (PID 或 Fuzzy): ").strip().lower()
        if control_choice in ['pid', 'fuzzy']:
            break
        else:
            print("无效选择，请输入 'PID' 或 'Fuzzy'。(Invalid choice, please enter 'PID' or 'Fuzzy'.)")

    # --- Get Inputs ---
    target_h2s = get_float_input("请输入 目标H2S浓度 (ppm): ")
    current_h2s = get_float_input("请输入 当前H2S浓度 (ppm): ")
    current_rpm = get_float_input("请输入 当前转速 (RPM): ")
    current_flow = get_float_input("请输入 当前脱硫液流量 (m³/h): ")

    # --- Initialize Controllers ---
    # Use settings similar to the simulation for consistency
    # Note: Kp for L is negative because increasing flow DECREASES concentration
    pid_L_config = PIDConfig(Kp=-0.25, Ki=-0.001, Kd=-0.015, output_limits=(-1.5, 1.5)) # Adjustment limits per step for Flow
    pid_L = PIDController(pid_L_config, setpoint=target_h2s)

    rpm_controller = None
    if control_choice == 'pid':
        # Note: Kp for RPM is negative because increasing RPM DECREASES concentration
        pid_rpm_config = PIDConfig(Kp=-0.8, Ki=-0.03, Kd=-0.6, output_limits=(-30, 30)) # Adjustment limits per step for RPM
        rpm_controller = PIDController(pid_rpm_config, setpoint=target_h2s)
    else: # fuzzy
        rpm_controller = FuzzyController(setpoint=target_h2s, output_limits=(-50, 50)) # Adjustment limits per step for RPM

    # --- Compute Adjustments ---
    # Use current time (or a fixed step) for calculation
    current_time = time.time() # Or just use 1.0 if absolute time isn't relevant

    # Reset controllers before computing for a clean calculation based only on current state
    pid_L.reset()
    rpm_controller.reset()

    # Compute adjustments based on the *current* error
    adj_L = pid_L.compute(current_h2s, current_time)
    adj_RPM = rpm_controller.compute(current_h2s, current_time)

    # --- Calculate Recommended Next Values ---
    # Apply adjustments and constraints (same as in the simulation)
    recommended_flow = max(0.0, min(current_flow + adj_L, 60.0)) # Flow range 0-60
    recommended_rpm = max(0.0, min(current_rpm + adj_RPM, 900.0)) # RPM range 0-900

    # --- Output Recommendations ---
    print("-" * 20)
    print(f"基于当前状态和目标值 {target_h2s} ppm (Based on current state and target {target_h2s} ppm):")
    print(f"  - 使用 {control_choice.upper()} 控制器 (Using {control_choice.upper()} controller)")
    print(f"  - 建议的下一步脱硫液流量调整量 (Recommended next liquid flow adjustment): {adj_L:+.2f} m³/h")
    print(f"  - 建议的下一步转速调整量 (Recommended next RPM adjustment): {adj_RPM:+.2f} RPM")
    print("-" * 20)
    print(f"  - 建议的下一步脱硫液流量 (Recommended next liquid flow rate): {recommended_flow:.2f} m³/h")
    print(f"  - 建议的下一步转速 (Recommended next RPM): {recommended_rpm:.2f} RPM")
    print("-" * 20)


# ========================
# 主程序入口 (Main Program Entry Point)
# ========================
def main():
    """Main function to run the interactive system."""
    while True:
        print("\n========== H2S 控制系统菜单 (H2S Control System Menu) ==========")
        print("1. 计算H2S出口浓度 (Calculate H2S Outlet Concentration)")
        print("2. 智能控制 (Intelligent Control - Recommend Adjustments)")
        print("3. 退出 (Exit)")
        print("==============================================================")

        choice = input("请选择一个选项 (1/2/3): ")

        if choice == '1':
            calculate_h2s_ui()
        elif choice == '2':
            intelligent_control_ui()
        elif choice == '3':
            print("正在退出程序... (Exiting program...)")
            break
        else:
            print("无效选择，请输入 1, 2, 或 3。(Invalid choice, please enter 1, 2, or 3.)")

        input("\n按 Enter 键继续... (Press Enter to continue...)") # Pause for user to read output

if __name__ == "__main__":
    main()
