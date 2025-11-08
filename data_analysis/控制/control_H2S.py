import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from math import erf, exp, sqrt, pi

class H2SHigeeSystem:
    def __init__(self, rpm, l_flow, tannic_conc, sodium_conc, ph):
        self.r1 = 42e-3
        self.r2 = 146e-3
        self.h = 20e-3
        self.P = 101.325
        self.H = 709.1735
        self.packing_area = 500
        self.packing_volume = np.pi * (self.r2**2 - self.r1**2) * self.h
        self.r = np.sqrt(self.r1 * self.r2)
        self.D_h2s = 1.8e-9
        self.g_flow = 2 / 3600

        self.rpm = rpm
        self.l_flow = l_flow
        self.tannic_conc = tannic_conc
        self.sodium_conc = sodium_conc
        self.ph = ph

        self.k_tannic = 0.05
        self.k_sodium = 0.2
        self.k_apparent = self.tannic_conc * self.k_tannic + self.sodium_conc * self.k_sodium

        self.flow_area = np.pi * (1.6e-3)**2
        self.l_speed = self.l_flow / self.flow_area

        self.y_in = 0.5
        self.concentration_ratio = (1 - self.y_in) / self.y_in
        self.packing_flow_factor = self.packing_area * np.pi * self.h * (self.r2**2 - self.r1**2) / self.g_flow

    def cal_t_new(self):
        t_new = (self.r2 - self.r1) / (0.02107 * (self.l_speed ** 0.2279) * ((self.r * self.rpm) ** 0.5448) * 31)
        return t_new

    def calculate_transfer_l(self, t_new):
        transfer_l = (sqrt(self.D_h2s * self.k_apparent) / t_new) * (
            t_new * erf(sqrt(self.k_apparent * t_new)) +
            sqrt((t_new / pi) / self.k_apparent) * exp(-self.k_apparent * t_new) +
            (0.5 / self.k_apparent) * erf(sqrt(self.k_apparent * t_new))
        )
        return transfer_l

    def compute_removal_efficiency(self):
        t_new = self.cal_t_new()
        transfer_l = self.calculate_transfer_l(t_new)

        transfer_y = 0.082 * 298.15 * 0.2700 * transfer_l

        yout = 1 / (self.concentration_ratio * np.exp(self.packing_flow_factor * transfer_y) + 1)
        yout = yout
        remove = (self.y_in - yout) / self.y_in
        return remove

# 定义模糊变量
error = ctrl.Antecedent(np.arange(-50, 51, 1), 'error')
delta_error = ctrl.Antecedent(np.arange(-10, 11, 1), 'delta_error')
rpm = ctrl.Consequent(np.arange(0, 1401, 1), 'rpm')
l_flow = ctrl.Consequent(np.arange(0, 0.01, 0.0001), 'l_flow')

# 定义隶属函数
error['negative'] = fuzz.trimf(error.universe, [-50, -50, 0])
error['zero'] = fuzz.trimf(error.universe, [-50, 0, 50])
error['positive'] = fuzz.trimf(error.universe, [0, 50, 50])

delta_error['negative'] = fuzz.trimf(delta_error.universe, [-10, -10, 0])
delta_error['zero'] = fuzz.trimf(delta_error.universe, [-10, 0, 10])
delta_error['positive'] = fuzz.trimf(delta_error.universe, [0, 10, 10])

rpm['low'] = fuzz.trimf(rpm.universe, [0, 0, 700])
rpm['medium'] = fuzz.trimf(rpm.universe, [0, 700, 1400])
rpm['high'] = fuzz.trimf(rpm.universe, [700, 1400, 1400])

l_flow['low'] = fuzz.trimf(l_flow.universe, [0, 0, 0.005])
l_flow['medium'] = fuzz.trimf(l_flow.universe, [0, 0.005, 0.01])
l_flow['high'] = fuzz.trimf(l_flow.universe, [0.005, 0.01, 0.01])

# 确保至少有一条规则能被激活
rule1 = ctrl.Rule(error['negative'] & delta_error['negative'], (rpm['high'], l_flow['high']))
rule2 = ctrl.Rule(error['negative'] & delta_error['zero'], (rpm['medium'], l_flow['medium']))
rule3 = ctrl.Rule(error['negative'] & delta_error['positive'], (rpm['low'], l_flow['low']))
rule4 = ctrl.Rule(error['zero'] & delta_error['negative'], (rpm['medium'], l_flow['medium']))
rule5 = ctrl.Rule(error['zero'] & delta_error['zero'], (rpm['medium'], l_flow['medium']))
rule6 = ctrl.Rule(error['zero'] & delta_error['positive'], (rpm['medium'], l_flow['medium']))
rule7 = ctrl.Rule(error['positive'] & delta_error['negative'], (rpm['low'], l_flow['low']))
rule8 = ctrl.Rule(error['positive'] & delta_error['zero'], (rpm['medium'], l_flow['medium']))
rule9 = ctrl.Rule(error['positive'] & delta_error['positive'], (rpm['high'], l_flow['high']))

# 创建控制系统和模拟器
removal_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
removal_sim = ctrl.ControlSystemSimulation(removal_ctrl)

# 模糊控制器测试
target_efficiency = 90  # 目标去除效率
previous_error = 0

# 初始条件
rpm_value = 1200
l_flow_value = 20e-3 / 3600
tannic_conc = 0.4
sodium_conc = 0.8
ph = 7.0

# 创建系统实例
system = H2SHigeeSystem(rpm_value, l_flow_value, tannic_conc, sodium_conc, ph)
current_efficiency = system.compute_removal_efficiency()

# 计算误差和误差变化率
error_value = target_efficiency - current_efficiency
delta_error_value = error_value - previous_error

# 输入模糊控制器
removal_sim.input['error'] = error_value
removal_sim.input['delta_error'] = delta_error_value
removal_sim.compute()

# 获取调整后的参数
adjusted_rpm = removal_sim.output['rpm']
adjusted_l_flow = removal_sim.output['l_flow']

print(f"Adjusted RPM: {adjusted_rpm}")
print(f"Adjusted Liquid Flow: {adjusted_l_flow}")

# 更新系统参数
adjusted_system = H2SHigeeSystem(adjusted_rpm, adjusted_l_flow, tannic_conc, sodium_conc, ph)
adjusted_efficiency = adjusted_system.compute_removal_efficiency()

print(f"Adjusted Removal Efficiency: {adjusted_efficiency}")

# 更新误差
previous_error = error_value
