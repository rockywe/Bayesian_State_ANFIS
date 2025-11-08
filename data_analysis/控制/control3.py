import tkinter as tk
from tkinter import ttk  # Themed widgets for better look
from tkinter import messagebox
from tkinter import scrolledtext # For displaying simulation results
import math
from dataclasses import dataclass
import time
import copy

# ==============================================================================
# Backend Logic (Copied from the previous script)
# ==============================================================================

# --- Core Process Calculation Model ---
def calculate_outlet_h2s(
        煤气进口流量_m3h, 进口煤气温度_C, 进口煤气压力_kPa,
        脱硫液流量_m3h, 脱硫液温度_C, 脱硫液压力_kPa,
        转速_RPM, 进口H2S浓度_ppm, L_exponent=0.6, RPM_exponent=0.8,
        G_exponent=-0.25, gas_velocity_factor=1.2, enhancement_factor=2.5,
        contact_time_base=0.8
):
    """Calculates the estimated H2S outlet concentration."""
    # Constants
    D_H2S = 1.8e-9
    H_H2S = 483.0
    alpha = 800
    R_gas = 0.0821
    liquid_density = 1100
    R_inner = 0.015
    R_outer = 0.85
    h_packing = 0.033

    try:
        # Basic validation for negative or None inputs
        inputs = {
            "煤气进口流量_m3h": 煤气进口流量_m3h, "进口煤气压力_kPa": 进口煤气压力_kPa,
            "脱硫液流量_m3h": 脱硫液流量_m3h, "脱硫液压力_kPa": 脱硫液压力_kPa,
            "转速_RPM": 转速_RPM, "进口H2S浓度_ppm": 进口H2S浓度_ppm
        }
        for name, v in inputs.items():
             if v is None or v < 0:
                  raise ValueError(f"输入参数 '{name}' 包含无效值 (负数或空值)。(Input parameter '{name}' contains invalid value (negative or None)).")

        # Unit Conversions and Intermediate Calculations
        G_m3s = 煤气进口流量_m3h / 3600
        T_gas = 进口煤气温度_C + 273.15
        P_total = 进口煤气压力_kPa / 101.325 if 进口煤气压力_kPa > 0 else 0
        y_in = 进口H2S浓度_ppm * 1e-6
        L_m3s = 脱硫液流量_m3h / 3600
        T_liquid = 脱硫液温度_C + 273.15
        R_avg = math.sqrt(R_inner * R_outer)
        omega = 转速_RPM * 2 * math.pi / 60

        # Handle edge cases where model might be invalid
        if G_m3s <= 0 or L_m3s <= 0 or 转速_RPM <= 0:
            # print("Warning: Zero flow or RPM. Returning inlet concentration.")
            return float(进口H2S浓度_ppm)

        centrifugal_g = (omega ** 2 * R_avg) / 9.81
        if centrifugal_g <= 0:
            # print("Warning: Zero centrifugal force. Returning inlet concentration.")
            return float(进口H2S浓度_ppm)

        # Mass Transfer Calculations (with checks for potential math errors)
        try:
            term_g = centrifugal_g ** (RPM_exponent * enhancement_factor)
            term_l = L_m3s ** (L_exponent * enhancement_factor)
            term_gas = G_m3s ** (G_exponent * enhancement_factor) # G_exponent is negative
        except ValueError as power_error:
             raise ValueError(f"计算指数时出错 (Error calculating exponent): {power_error}")

        kLa = 0.024 * enhancement_factor * term_g * term_l * term_gas
        if alpha == 0: raise ValueError("有效比表面积 alpha 不能为零。(Effective specific surface area alpha cannot be zero.)")
        kL = kLa / alpha

        cross_area = math.pi * (R_outer ** 2 - R_inner ** 2)
        if cross_area <= 0:
             # print("Warning: Invalid cross-sectional area. Returning inlet concentration.")
             return float(进口H2S浓度_ppm)

        liquid_velocity = L_m3s / cross_area
        gas_velocity = G_m3s / cross_area
        combined_velocity = (liquid_velocity + gas_velocity_factor * gas_velocity * enhancement_factor)
        if combined_velocity <= 0:
             # print("Warning: Zero combined velocity. Returning inlet concentration.")
             return float(进口H2S浓度_ppm)

        residence_time = contact_time_base * h_packing / combined_velocity
        NTU = kL * alpha * residence_time
        NTU = max(0.0, min(NTU, 50.0)) # Bounding NTU

        denominator = (1 + NTU / (5 * enhancement_factor))
        if denominator <= 1e-9: # Check for near-zero or negative denominator
            # print(f"Warning: Denominator near zero ({denominator}). Using fallback calculation.")
            # Fallback or alternative calculation if denominator is problematic
            y_out = y_in * math.exp(-NTU)
        else:
            try:
                y_out = y_in * math.exp(-NTU / denominator)
            except OverflowError:
                 # Handle cases where exp argument becomes too large negatively
                 y_out = 0.0 # Concentration effectively becomes zero

        # Final empirical calculation and bounding
        outlet_ppm = y_out * 1e6 / 40
        return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm * 1.2)) # Cap at slightly above inlet

    except ValueError as ve:
        # Display specific value errors to the user
        messagebox.showerror("计算错误 (Calculation Error)", f"数值错误 (Numerical error): {ve}")
        return None # Indicate error
    except Exception as e:
        # Catch any other unexpected errors
        messagebox.showerror("计算错误 (Calculation Error)", f"计算过程中发生意外错误 (Unexpected error during calculation): {e}")
        return None # Indicate error


# --- PID Controller Class ---
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
        # Robust delta time calculation
        if self.last_time is None: dt = 1.0 # Assume 1s for first step
        else: dt = current_time - self.last_time
        # Prevent issues with non-positive or excessively large dt
        if dt <= 1e-6 or dt > 100: dt = 1.0

        self.last_time = current_time
        error = self.setpoint - current_value

        # Integral term (Trapezoidal rule)
        new_integral = self.integral + (error + self.prev_error) * dt / 2.0

        # Derivative term (Backward difference)
        derivative = (error - self.prev_error) / dt

        # PID output
        output = (self.config.Kp * error + self.config.Ki * new_integral + self.config.Kd * derivative)

        # Output clamping and Anti-windup
        min_limit, max_limit = self.config.output_limits
        if output > max_limit:
            output = max_limit
            # If output is clamped, only allow integral to decrease if error has opposite sign
            if error * output > 0: new_integral = self.integral
        elif output < min_limit:
            output = min_limit
            # If output is clamped, only allow integral to increase if error has opposite sign
            if error * output > 0: new_integral = self.integral

        self.integral = new_integral # Update integral state
        self.prev_error = error # Store error for next iteration
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

# --- Fuzzy Controller Class ---
class FuzzyController:
    # (Fuzzy Controller code remains the same as in the previous version)
    def __init__(self, setpoint, output_limits):
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.prev_error = 0.0
        self.last_time = None
        self.prev_dRPM_adj = 0.0
        self.e_universe = (-100, 100)
        self.e_mf = {'NB': lambda e: self._trimf(e, -100, -100, -30),'NS': lambda e: self._trimf(e, -60, -30, 0),'Z': lambda e: self._trimf(e, -10, 0, 10),'PS': lambda e: self._trimf(e, 0, 30, 60),'PB': lambda e: self._trimf(e, 30, 100, 100)}
        self.de_universe = (-30, 30)
        self.de_mf = {'NB': lambda de: self._trimf(de, -30, -30, -10),'NS': lambda de: self._trimf(de, -20, -10, 0),'Z': lambda de: self._trimf(de, -3, 0, 3),'PS': lambda de: self._trimf(de, 0, 10, 20),'PB': lambda de: self._trimf(de, 10, 30, 30)}
        self.dRPM_adj_universe = (-100, 100)
        self.dRPM_adj_mf = {'NB': -90, 'NS': -40, 'Z': 0, 'PS': 40, 'PB': 90}
        self.rules = {('NB', 'NB'): 'PB', ('NB', 'NS'): 'PB', ('NB', 'Z'): 'PB', ('NB', 'PS'): 'PS', ('NB', 'PB'): 'Z',('NS', 'NB'): 'PB', ('NS', 'NS'): 'PS', ('NS', 'Z'): 'PS', ('NS', 'PS'): 'Z', ('NS', 'PB'): 'NS',('Z', 'NB'): 'PS', ('Z', 'NS'): 'Z', ('Z', 'Z'): 'Z', ('Z', 'PS'): 'Z', ('Z', 'PB'): 'NS',('PS', 'NB'): 'NS', ('PS', 'NS'): 'Z', ('PS', 'Z'): 'NS', ('PS', 'PS'): 'NB', ('PS', 'PB'): 'NB',('PB', 'NB'): 'Z', ('PB', 'NS'): 'NB', ('PB', 'Z'): 'NB', ('PB', 'PS'): 'NB', ('PB', 'PB'): 'NB'}

    def _trimf(self, x, a, b, c):
        try:
            a, b, c = sorted((a, b, c))
            if a == c: return 1.0 if x == a else 0.0
            val1 = (x - a) / (b - a) if b != a else (1.0 if x >= b else 0.0)
            val2 = (c - x) / (c - b) if c != b else (1.0 if x <= b else 0.0)
            return max(0.0, min(val1 if b != a else (1.0 if x >= b else 0.0), val2 if c != b else (1.0 if x <= b else 0.0)))
        except ZeroDivisionError: return 1.0 if a <= x <= c else 0.0

    def _fuzzify(self, value, mf_dict, universe):
        min_u, max_u = universe
        clamped_value = max(min_u, min(value, max_u))
        return {term: mf(clamped_value) for term, mf in mf_dict.items()}

    def _inference(self, fuzzified_e, fuzzified_de):
        rule_strengths = {}
        for (e_term, de_term), output_term in self.rules.items():
            strength = min(fuzzified_e.get(e_term, 0), fuzzified_de.get(de_term, 0))
            if strength > 0:
                if output_term not in rule_strengths or strength > rule_strengths[output_term]:
                    rule_strengths[output_term] = strength
        return rule_strengths

    def _defuzzify(self, rule_strengths):
        numerator = 0
        denominator = 0
        if not rule_strengths: return 0.0
        for term, strength in rule_strengths.items():
            if term in self.dRPM_adj_mf:
                numerator += strength * self.dRPM_adj_mf[term]
                denominator += strength
        return numerator / denominator if denominator != 0 else 0

    def compute(self, current_value, current_time):
        if self.last_time is None: dt = 1.0
        else: dt = current_time - self.last_time
        if dt <= 1e-6 or dt > 100: dt = 1.0 # Robust dt
        self.last_time = current_time
        error = self.setpoint - current_value
        de = (error - self.prev_error) / dt
        self.prev_error = error
        fuzzified_e = self._fuzzify(error, self.e_mf, self.e_universe)
        fuzzified_de = self._fuzzify(de, self.de_mf, self.de_universe)
        rule_strengths = self._inference(fuzzified_e, fuzzified_de)
        dRPM_adj = self._defuzzify(rule_strengths)
        min_limit, max_limit = self.output_limits
        dRPM_adj = max(min_limit, min(dRPM_adj, max_limit))
        # Apply filtering as per original fuzzy simulation logic
        filtered_dRPM_adj = 0.9 * dRPM_adj + 0.1 * self.prev_dRPM_adj
        self.prev_dRPM_adj = filtered_dRPM_adj
        return filtered_dRPM_adj # Return filtered value

    def reset(self):
        self.prev_error = 0.0
        self.last_time = None
        self.prev_dRPM_adj = 0.0

# --- Simulation Function ---
def run_prediction_simulation(initial_params, target_h2s, rpm_control_type, duration_s=600, dt_s=5, progress_callback=None):
    """Runs a simulation loop, calling progress_callback periodically."""
    process_params = copy.deepcopy(initial_params)
    h2s_history = []
    log_output = [] # Store log messages

    log_output.append(f"--- 运行 {rpm_control_type.upper()} 全程预测 ({duration_s} 秒) ---")
    log_output.append(f"--- Running {rpm_control_type.upper()} Full Prediction ({duration_s}s) ---")

    # Initialize controllers
    pid_L_config = PIDConfig(Kp=-0.25, Ki=-0.001, Kd=-0.015, output_limits=(-1.5, 1.5))
    pid_L = PIDController(pid_L_config, setpoint=target_h2s)
    if rpm_control_type == 'pid':
        pid_rpm_config = PIDConfig(Kp=-0.8, Ki=-0.03, Kd=-0.6, output_limits=(-30, 30))
        rpm_controller = PIDController(pid_rpm_config, setpoint=target_h2s)
    elif rpm_control_type == 'fuzzy':
        rpm_controller = FuzzyController(setpoint=target_h2s, output_limits=(-50, 50))
    else:
        raise ValueError("无效的转速控制类型。(Invalid rpm_control_type.)")

    current_time_sim = 0
    pid_L.reset()
    rpm_controller.reset()

    try:
        # Calculate initial concentration
        initial_conc = calculate_outlet_h2s(**process_params)
        if initial_conc is None: raise ValueError("无法计算初始浓度。(Could not calculate initial concentration.)")
        h2s_history.append({'time': 0, 'conc': initial_conc})
        log_line = f"时间 0s: H2S = {initial_conc:.2f} ppm, 转速 = {process_params['转速_RPM']:.1f} RPM, 流量 = {process_params['脱硫液流量_m3h']:.1f} m³/h"
        log_output.append(log_line)
        if progress_callback: progress_callback(log_line) # Initial state

        # Simulation loop
        for t_step in range(dt_s, duration_s + dt_s, dt_s):
            current_time_sim += dt_s
            # Calculate concentration based on *current* parameters before adjustment
            current_conc = calculate_outlet_h2s(**process_params)
            if current_conc is None: raise ValueError(f"无法在时间 {t_step}s 计算浓度。(Could not calculate concentration at time {t_step}s.)")
            h2s_history.append({'time': t_step, 'conc': current_conc})

            # Compute adjustments based on current concentration
            adj_L = pid_L.compute(current_conc, current_time_sim)
            adj_RPM = rpm_controller.compute(current_conc, current_time_sim) # Fuzzy compute includes filtering

            # Apply adjustments and constraints for the *next* state
            process_params["脱硫液流量_m3h"] = max(0.0, min(process_params["脱硫液流量_m3h"] + adj_L, 60.0))
            process_params["转速_RPM"] = max(0.0, min(process_params["转速_RPM"] + adj_RPM, 900.0))

            # Log progress periodically
            if t_step % 60 == 0 or t_step == duration_s: # Log every minute and at the end
                # Log the state *after* adjustments were applied (which will be used in next calc)
                log_line = f"时间 {t_step}s: H2S = {current_conc:.2f} ppm, 转速 = {process_params['转速_RPM']:.1f} RPM, 流量 = {process_params['脱硫液流量_m3h']:.1f} m³/h"
                log_output.append(log_line)
                if progress_callback: progress_callback(log_line) # Update GUI

        # Calculate final concentration based on the very last set of parameters
        final_conc = calculate_outlet_h2s(**process_params)
        if final_conc is None: final_conc = h2s_history[-1]['conc'] # Use last calculated if final fails

        # Log final results
        log_output.append(f"预测结束 (Prediction End): 时间 {duration_s}s")
        log_output.append(f"  - 最终预测 H2S 浓度 (Final Predicted H2S): {final_conc:.2f} ppm")
        log_output.append(f"  - 最终预测 转速 (Final Predicted RPM): {process_params['转速_RPM']:.2f} RPM")
        log_output.append(f"  - 最终预测 脱硫液流量 (Final Predicted Liquid Flow): {process_params['脱硫液流量_m3h']:.2f} m³/h")
        if progress_callback:
            # Ensure final lines are sent to the log
            progress_callback(log_output[-3])
            progress_callback(log_output[-2])
            progress_callback(log_output[-1])

        return process_params, h2s_history, "\n".join(log_output)

    except Exception as e:
        error_msg = f"模拟出错 (Simulation Error): {e}"
        log_output.append(error_msg)
        if progress_callback: progress_callback(error_msg)
        # Return indicating failure
        return None, [], "\n".join(log_output)


# ==============================================================================
# GUI Application Class
# ==============================================================================
class H2SControlApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("H2S 控制系统 (H2S Control System)")
        self.geometry("850x700") # Increased height slightly for padding

        # --- Style ---
        self.style = ttk.Style(self)
        self.style.theme_use('clam') # Or 'alt', 'default', 'classic'

        # --- Main Notebook (Tabs) ---
        self.notebook = ttk.Notebook(self)

        # --- Tab 1: Calculate Outlet Concentration ---
        self.tab_calculate = ttk.Frame(self.notebook, padding="15") # Increased padding
        self.notebook.add(self.tab_calculate, text='计算出口浓度 (Calculate Outlet Conc.)')
        self.create_calculate_tab()

        # --- Tab 2: Intelligent Control ---
        self.tab_control = ttk.Frame(self.notebook, padding="15") # Increased padding
        self.notebook.add(self.tab_control, text='智能控制与预测 (Intelligent Control & Prediction)')
        self.create_control_tab()

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10) # Add padding around notebook

    # --- Helper to create labeled entry ---
    def create_labeled_entry(self, parent, text, row, col, default_value="", width=15):
        """Creates a Label and an Entry widget. Returns the StringVar and the Entry widget."""
        ttk.Label(parent, text=text).grid(row=row, column=col, padx=5, pady=5, sticky="w")
        entry_var = tk.StringVar(value=default_value)
        entry_widget = ttk.Entry(parent, textvariable=entry_var, width=width) # Assign widget
        entry_widget.grid(row=row, column=col + 1, padx=5, pady=5, sticky="ew")
        return entry_var, entry_widget # Return both

    # --- Helper to get float from entry ---
    def get_float_from_entry(self, entry_var, var_name="输入"):
        """Safely gets a float value from a Tkinter StringVar."""
        try:
            value_str = entry_var.get()
            if not value_str: # Handle empty input
                 # Allow empty for non-critical fields if needed, or raise error
                 raise ValueError("值不能为空。(Value cannot be empty.)")
            return float(value_str)
        except ValueError:
            messagebox.showerror("输入错误 (Input Error)", f"'{var_name}' 的值无效。请输入一个数字。(Invalid value for '{var_name}'. Please enter a number.)")
            return None # Indicate error

    # --- Tab 1 Creation ---
    def create_calculate_tab(self):
        frame = self.tab_calculate
        frame.columnconfigure(1, weight=1) # Allow entry column to expand

        # Input Fields - Store only the variables, widgets not needed later here
        self.calc_entries = {}
        self.calc_entries['煤气进口流量_m3h'], _ = self.create_labeled_entry(frame, "煤气进口流量 (m³/h):", 0, 0, "1500")
        self.calc_entries['进口煤气温度_C'], _ = self.create_labeled_entry(frame, "进口煤气温度 (℃):", 1, 0, "40")
        self.calc_entries['进口煤气压力_kPa'], _ = self.create_labeled_entry(frame, "进口煤气压力 (kPa):", 2, 0, "20")
        self.calc_entries['脱硫液流量_m3h'], _ = self.create_labeled_entry(frame, "脱硫液流量 (m³/h):", 3, 0, "30")
        self.calc_entries['脱硫液温度_C'], _ = self.create_labeled_entry(frame, "脱硫液温度 (℃):", 4, 0, "35")
        self.calc_entries['脱硫液压力_kPa'], _ = self.create_labeled_entry(frame, "脱硫液压力 (kPa):", 5, 0, "40")
        self.calc_entries['转速_RPM'], _ = self.create_labeled_entry(frame, "转速 (RPM):", 6, 0, "450")
        self.calc_entries['进口H2S浓度_ppm'], _ = self.create_labeled_entry(frame, "进口H2S浓度 (ppm):", 7, 0, "1000")

        # Calculate Button
        calc_button = ttk.Button(frame, text="计算 (Calculate)", command=self.perform_calculation)
        calc_button.grid(row=8, column=0, columnspan=2, pady=20) # Increased pady

        # Result Display
        ttk.Label(frame, text="计算结果 (Result):").grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.calc_result_var = tk.StringVar(value="--")
        result_label = ttk.Label(frame, textvariable=self.calc_result_var, font=("Arial", 12, "bold"), anchor="w") # Anchor left
        result_label.grid(row=9, column=1, padx=5, pady=5, sticky="ew")

    # --- Tab 2 Creation ---
    def create_control_tab(self):
        frame = self.tab_control
        # Configure column weights for responsiveness
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=1)
        frame.rowconfigure(5, weight=1) # Allow log area row to expand

        # --- Input Section ---
        input_frame = ttk.LabelFrame(frame, text="当前状态与目标 (Current State & Target)", padding="10")
        input_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(3, weight=1)

        self.control_entries = {}
        self.control_entries['target_h2s'], _ = self.create_labeled_entry(input_frame, "目标H2S浓度 (ppm):", 0, 0, "20")
        self.control_entries['current_h2s'], _ = self.create_labeled_entry(input_frame, "当前H2S浓度 (ppm):", 1, 0)
        self.control_entries['current_rpm'], _ = self.create_labeled_entry(input_frame, "当前转速 (RPM):", 0, 2, "450")
        self.control_entries['current_flow'], _ = self.create_labeled_entry(input_frame, "当前脱硫液流量 (m³/h):", 1, 2, "30")

        # --- Other Parameters Section (for simulation) ---
        sim_params_frame = ttk.LabelFrame(frame, text="其他工艺参数 (用于模拟) Other Process Parameters (for Simulation)", padding="10")
        sim_params_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        sim_params_frame.columnconfigure(1, weight=1)
        sim_params_frame.columnconfigure(3, weight=1)

        self.sim_entries = {} # Store variables only
        self.sim_entries['煤气进口流量_m3h'], _ = self.create_labeled_entry(sim_params_frame, "煤气进口流量 (m³/h):", 0, 0, "1500")
        self.sim_entries['进口煤气温度_C'], _ = self.create_labeled_entry(sim_params_frame, "进口煤气温度 (℃):", 1, 0, "40")
        self.sim_entries['进口煤气压力_kPa'], _ = self.create_labeled_entry(sim_params_frame, "进口煤气压力 (kPa):", 2, 0, "20")
        self.sim_entries['脱硫液温度_C'], _ = self.create_labeled_entry(sim_params_frame, "脱硫液温度 (℃):", 0, 2, "35")
        self.sim_entries['脱硫液压力_kPa'], _ = self.create_labeled_entry(sim_params_frame, "脱硫液压力 (kPa):", 1, 2, "40")
        self.sim_entries['进口H2S浓度_ppm'], _ = self.create_labeled_entry(sim_params_frame, "进口H2S浓度 (ppm):", 2, 2, "1000")


        # --- Control Mode Selection ---
        mode_frame = ttk.LabelFrame(frame, text="控制/预测模式 (Control/Prediction Mode)", padding="10")
        mode_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=10, sticky="ew")

        self.control_mode_var = tk.StringVar(value="pid_next") # Default selection

        modes = [
            ("PID 下一步预测 (PID Next Step)", "pid_next"),
            ("Fuzzy 下一步预测 (Fuzzy Next Step)", "fuzzy_next"),
            ("最优化 下一步预测 (Optimization Next Step) (RPM: 700-800)", "opt_next"),
            ("PID 全程预测 (PID Full Prediction) (10 min)", "pid_full"),
            ("Fuzzy 全程预测 (Fuzzy Full Prediction) (10 min)", "fuzzy_full")
        ]

        for i, (text, mode_val) in enumerate(modes):
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.control_mode_var, value=mode_val)
            # Grid radio buttons more compactly if needed
            rb.grid(row=i, column=0, sticky="w", padx=5, pady=2)

        # --- Action Button ---
        predict_button = ttk.Button(frame, text="执行预测/推荐 (Run Prediction/Recommendation)", command=self.perform_control_action)
        predict_button.grid(row=3, column=0, columnspan=4, pady=15)

        # --- Results Display ---
        result_frame = ttk.LabelFrame(frame, text="预测/推荐结果 (Prediction/Recommendation Result)", padding="10")
        result_frame.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        result_frame.columnconfigure(1, weight=1)
        result_frame.columnconfigure(3, weight=1)

        self.control_result_vars = {} # Store StringVars
        self.control_result_widgets = {} # Store Entry widgets

        # Create and store both var and widget
        var, widget = self.create_labeled_entry(result_frame, "建议流量调整量 (Adj. Flow):", 0, 0, "--", width=12)
        self.control_result_vars['adj_L'] = var
        self.control_result_widgets['adj_L'] = widget

        var, widget = self.create_labeled_entry(result_frame, "建议转速调整量 (Adj. RPM):", 1, 0, "--", width=12)
        self.control_result_vars['adj_RPM'] = var
        self.control_result_widgets['adj_RPM'] = widget

        var, widget = self.create_labeled_entry(result_frame, "建议下一步流量 (Next Flow):", 0, 2, "--", width=12)
        self.control_result_vars['next_flow'] = var
        self.control_result_widgets['next_flow'] = widget

        var, widget = self.create_labeled_entry(result_frame, "建议下一步转速 (Next RPM):", 1, 2, "--", width=12)
        self.control_result_vars['next_rpm'] = var
        self.control_result_widgets['next_rpm'] = widget

        # Disable result entries initially using the stored widgets
        for widget in self.control_result_widgets.values():
            widget.config(state='readonly')


        # --- Simulation Log Area ---
        log_frame = ttk.LabelFrame(frame, text="全程预测日志 (Full Prediction Log)", padding="10")
        # Changed row to 5 to place it below results
        log_frame.grid(row=5, column=0, columnspan=4, padx=5, pady=10, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, state='disabled', font=("Courier New", 9)) # Use monospace font
        self.log_text.grid(row=0, column=0, sticky="nsew")


    # --- Action Handlers ---
    def perform_calculation(self):
        """Handles the calculate button click."""
        params = {}
        valid_input = True
        # Retrieve values using the stored StringVars
        for key, var in self.calc_entries.items():
            value = self.get_float_from_entry(var, key)
            if value is None:
                valid_input = False
                break
            params[key] = value

        if not valid_input:
            self.calc_result_var.set("输入错误 (Input Error)")
            return

        # Call the backend calculation function
        outlet_concentration = calculate_outlet_h2s(**params)

        # Update the result label
        if outlet_concentration is not None:
            self.calc_result_var.set(f"{outlet_concentration:.2f} ppm")
        else:
            # Error message is shown by calculate_outlet_h2s via messagebox
            self.calc_result_var.set("计算失败 (Calculation Failed)")

    def update_log(self, message):
        """Appends a message to the log area in a thread-safe way."""
        # Ensure GUI updates happen on the main thread if using threading later
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END) # Scroll to the bottom
        self.log_text.config(state='disabled')
        self.update_idletasks() # Force GUI update immediately

    def clear_results_and_log(self):
        """Clears previous results and log."""
        # Clear result fields by setting their StringVars
        for var in self.control_result_vars.values():
             var.set("--")

        # Clear log area
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')


    def perform_control_action(self):
        """Handles the predict/recommend button click."""
        self.clear_results_and_log() # Clear previous output

        # --- Get Common Inputs ---
        # Use stored StringVars to get values
        target_h2s = self.get_float_from_entry(self.control_entries['target_h2s'], "目标H2S浓度")
        current_h2s = self.get_float_from_entry(self.control_entries['current_h2s'], "当前H2S浓度")
        current_rpm = self.get_float_from_entry(self.control_entries['current_rpm'], "当前转速")
        current_flow = self.get_float_from_entry(self.control_entries['current_flow'], "当前脱硫液流量")

        if None in [target_h2s, current_h2s, current_rpm, current_flow]:
            return # Error message already shown

        # --- Get Simulation Parameters ---
        current_process_params = {}
        valid_sim_params = True
        # Get params needed for calculate_outlet_h2s from sim_entries StringVars
        for key, var in self.sim_entries.items():
            value = self.get_float_from_entry(var, key)
            if value is None:
                valid_sim_params = False
                break
            current_process_params[key] = value

        if not valid_sim_params:
            return

        # Add the control-specific current values from control_entries
        current_process_params["转速_RPM"] = current_rpm
        current_process_params["脱硫液流量_m3h"] = current_flow
        # Ensure model parameters have defaults if not in GUI
        current_process_params.setdefault("L_exponent", 0.6)
        current_process_params.setdefault("RPM_exponent", 0.8)
        current_process_params.setdefault("G_exponent", -0.25)
        current_process_params.setdefault("gas_velocity_factor", 1.2)
        current_process_params.setdefault("enhancement_factor", 2.5)
        current_process_params.setdefault("contact_time_base", 0.8)


        # --- Determine Action based on Mode ---
        mode = self.control_mode_var.get()

        # --- Enable result entries before updating ---
        for widget in self.control_result_widgets.values():
            widget.config(state='normal')

        # --- Handle Full Prediction Modes ---
        if mode == "pid_full" or mode == "fuzzy_full":
            rpm_ctrl_type = 'pid' if mode == "pid_full" else 'fuzzy'
            self.update_log("开始全程预测... (Starting full prediction...)")

            # Run simulation (consider threading for long simulations to avoid freezing GUI)
            final_params, _, log_text = run_prediction_simulation(
                current_process_params, target_h2s, rpm_ctrl_type,
                duration_s=600, dt_s=5, progress_callback=self.update_log
            )

            # Display final predicted state in the result fields
            if final_params:
                 # Recalculate final concentration for display consistency
                 final_conc = calculate_outlet_h2s(**final_params)
                 self.control_result_vars['adj_L'].set("N/A (全程)")
                 self.control_result_vars['adj_RPM'].set("N/A (全程)")
                 self.control_result_vars['next_flow'].set(f"{final_params['脱硫液流量_m3h']:.2f}")
                 self.control_result_vars['next_rpm'].set(f"{final_params['转速_RPM']:.2f}")
            else:
                 # Error logged by simulation function
                 messagebox.showerror("模拟错误", "全程预测模拟失败。(Full prediction simulation failed.)")
                 # Clear results if sim failed
                 self.control_result_vars['adj_L'].set("--")
                 self.control_result_vars['adj_RPM'].set("--")
                 self.control_result_vars['next_flow'].set("--")
                 self.control_result_vars['next_rpm'].set("--")


        # --- Handle Next Step Prediction Modes ---
        else: # pid_next, fuzzy_next, opt_next
            control_type_backend = 'pid' # Default for pid_next and opt_next
            if mode == "fuzzy_next":
                control_type_backend = 'fuzzy'

            is_optimization = (mode == "opt_next")

            # Initialize controllers
            pid_L_config = PIDConfig(Kp=-0.25, Ki=-0.001, Kd=-0.015, output_limits=(-1.5, 1.5))
            pid_L = PIDController(pid_L_config, setpoint=target_h2s)

            rpm_controller = None
            if control_type_backend == 'pid':
                pid_rpm_config = PIDConfig(Kp=-0.8, Ki=-0.03, Kd=-0.6, output_limits=(-30, 30))
                rpm_controller = PIDController(pid_rpm_config, setpoint=target_h2s)
            else: # fuzzy
                rpm_controller = FuzzyController(setpoint=target_h2s, output_limits=(-50, 50))

            # Compute adjustments
            current_time = time.time()
            pid_L.reset()
            rpm_controller.reset()
            try:
                # Ensure current_h2s is valid before passing to controllers
                if current_h2s is None: raise ValueError("当前H2S浓度无效。(Current H2S concentration is invalid.)")
                adj_L = pid_L.compute(current_h2s, current_time)
                adj_RPM = rpm_controller.compute(current_h2s, current_time)
            except Exception as compute_error:
                 messagebox.showerror("控制计算错误", f"计算调整量时出错: {compute_error}")
                 adj_L, adj_RPM = 0.0, 0.0 # Default to no adjustment on error


            # Calculate recommended next values
            recommended_flow = max(0.0, min(current_flow + adj_L, 60.0))
            recommended_rpm = max(0.0, min(current_rpm + adj_RPM, 900.0))
            rpm_clamped_info = ""

            # Apply optimization clamping if needed
            if is_optimization:
                rpm_lower_bound = 700.0
                rpm_upper_bound = 800.0
                if recommended_rpm < rpm_lower_bound:
                    recommended_rpm = rpm_lower_bound
                    rpm_clamped_info = " (已约束 Clamped to 700)"
                    self.update_log("提示: 推荐转速已调整至下限 700 RPM。(Info: Recommended RPM clamped to lower bound 700 RPM.)")
                elif recommended_rpm > rpm_upper_bound:
                    recommended_rpm = rpm_upper_bound
                    rpm_clamped_info = " (已约束 Clamped to 800)"
                    self.update_log("提示: 推荐转速已调整至上限 800 RPM。(Info: Recommended RPM clamped to upper bound 800 RPM.)")

            # Update result fields using StringVars
            self.control_result_vars['adj_L'].set(f"{adj_L:+.2f}")
            self.control_result_vars['adj_RPM'].set(f"{adj_RPM:+.2f}")
            self.control_result_vars['next_flow'].set(f"{recommended_flow:.2f}")
            self.control_result_vars['next_rpm'].set(f"{recommended_rpm:.2f}{rpm_clamped_info}")

            self.update_log("下一步预测完成。(Next step prediction complete.)")

        # --- Disable result entries again after updating ---
        for widget in self.control_result_widgets.values():
            widget.config(state='readonly')


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    app = H2SControlApp()
    app.mainloop()
