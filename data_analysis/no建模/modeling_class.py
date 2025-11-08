import numpy as np
from scipy.special import erf
from modeling_fuzzy import FuzzyModel

# 需要输入4个参数
class HigeeProcess:
    # 常量定义
    r1 = 42e-3
    r2 = 146e-3
    h = 20e-3
    P = 101.325
    H = 709.1735
    packing_area = 500
    packing_volume = np.pi * (r2 ** 2 - r1 ** 2) * h
    r = np.sqrt(r1 * r2)
    D_no = 2.21e-9
    g_flow = 2 / 3600

    def __init__(self, rpm, l_flow, c_h2o2, ph):
        self.y_in = 0.0005
        self.rpm = rpm
        self.ph = ph
        self.l_flow = l_flow
        self.c_h2o2 = c_h2o2
        self.flow_area = np.pi * (1.6e-3) ** 2
        self.l_speed = self.l_flow / self.flow_area
        self.k_noooh = 103.4
        self.al = 10 ** (11.73 - self.ph)
        self.a = 1 / (1 + self.al)
        self.ooh = self.c_h2o2 * self.a
        self.k_apparent = self.ooh * self.k_noooh
        self.concentration_ratio = (1 - self.y_in) / self.y_in
        self.packing_flow_factor = self.packing_area * np.pi * self.h * (self.r2 ** 2 - self.r1 ** 2) / self.g_flow

    def calculate_t_new(self):
        t_new = (self.r2 - self.r1) / (0.02107 * (self.l_speed ** 0.2279) * ((self.r * self.rpm) ** 0.5448) * 31)
        return t_new

    def calculate_transfer_l(self, t_new):
        transfer_l = (np.sqrt(self.D_no * self.k_apparent) / t_new) * (
                t_new * erf(np.sqrt(self.k_apparent * t_new)) +
                np.sqrt((t_new / np.pi) / self.k_apparent) * np.exp(-self.k_apparent * t_new) +
                (0.5 / self.k_apparent) * erf(np.sqrt(self.k_apparent * t_new))
        )
        return transfer_l

    def run_simulation(self):
        t_new = self.calculate_t_new()
        transfer_l = self.calculate_transfer_l(t_new)
        transfer_y = 0.082 * 298.15 * 0.2700 * transfer_l
        yout = 1 / (self.concentration_ratio * np.exp(self.packing_flow_factor * transfer_y) + 1)
        remove = (self.y_in - yout) / self.y_in
        return yout, remove

rpm=1200
l_flow=20e-3 / 3600
c_h2o2=0.4
ph=12.01

# 实例化类并运行模拟
process = HigeeProcess(rpm=rpm, l_flow=l_flow, c_h2o2=c_h2o2, ph=ph)
yout, remove = process.run_simulation()

fuzzy_model = FuzzyModel(pH=ph, rpm=rpm, ch2o2=c_h2o2, error=remove-0.96)
fuzzy_prediction = fuzzy_model.predict(ph, rpm, c_h2o2)

print("出口ppm(yout):", yout)
print("去除效率 (remove):", remove)
print("模糊模型预测 (fuzzy_prediction):", fuzzy_prediction)