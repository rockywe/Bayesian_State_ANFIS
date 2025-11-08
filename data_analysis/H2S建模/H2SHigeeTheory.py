import numpy as np
from scipy.special import erf
from DataHandler import DataHandler

class H2SHigeeTheory:
    def __init__(self, rpm, l_flow, tannic_conc, sodium_conc, ph):
    # def __init__(self, rpm=1200, l_flow=20e-3 / 3600, tannic_conc=0.4, sodium_conc=0.8, ph=7.0):
        # Constants
        self.r1 = 42e-3  # 内径，单位为米
        self.r2 = 146e-3  # 外径，单位为米
        self.h = 20e-3  # 高度，单位为米
        self.P = 101.325  # 压力，单位为千帕
        self.H = 709.1735  # Henry常数，单位为 KPa*m3/mol
        self.packing_area = 500  # 填料比表面积，单位为 m^2/m^3
        self.packing_volume = np.pi * (self.r2**2 - self.r1**2) * self.h  # 填料体积，单位为立方米
        self.r = np.sqrt(self.r1 * self.r2)  # 几何平均半径，单位为米
        self.D_h2s = 1.8e-9  # H2S在溶液中的扩散系数，单位为 m^2/s
        self.g_flow = 2 / 3600  # 气体流量，单位为立方米每秒

        # Variables
        self.rpm = rpm  # 转速
        self.l_flow = l_flow  # 液体流量
        self.tannic_conc = tannic_conc  # 单宁酸浓度，单位为 mol/L
        self.sodium_conc = sodium_conc  # 偏钒酸钠的浓度，单位为 mol/L
        self.ph = ph  # 假设中性pH值

        # 表观速率常数
        self.k_tannic = 0.05  # 单宁酸的表观速率常数
        self.k_sodium = 0.2  # 偏钒酸钠的表观速率常数
        self.k_apparent = self.tannic_conc * self.k_tannic + self.sodium_conc * self.k_sodium

        # 流速相关计算
        self.flow_area = np.pi * (1.6e-3)**2  # 横截面积
        self.l_speed = self.l_flow / self.flow_area  # 液相流速

        # 其他常量
        self.y_in = 0.5  # 初始H2S浓度
        self.concentration_ratio = (1 - self.y_in) / self.y_in  # 浓度比例
        self.packing_flow_factor = self.packing_area * np.pi * self.h * (self.r2**2 - self.r1**2) / self.g_flow  # 用来算出口体积

    def cal_t_new(self):
        t_new = (self.r2 - self.r1) / (0.02107 * (self.l_speed ** 0.2279) * ((self.r * self.rpm) ** 0.5448) * 31)
        return t_new

    def calculate_transfer_l(self, t_new):
        transfer_l = (np.sqrt(self.D_h2s * self.k_apparent) / t_new) * (
            t_new * erf(np.sqrt(self.k_apparent * t_new)) +
            np.sqrt((t_new / np.pi) / self.k_apparent) * np.exp(-self.k_apparent * t_new) +
            (0.5 / self.k_apparent) * erf(np.sqrt(self.k_apparent * t_new))
        )
        return transfer_l

    def compute_removal_efficiency(self):
        t_new = self.cal_t_new()
        transfer_l = self.calculate_transfer_l(t_new)

        # 气液传质系数
        transfer_y = 0.082 * 298.15 * 0.2700 * transfer_l

        # 出口浓度
        yout = 1 / (self.concentration_ratio * np.exp(self.packing_flow_factor * transfer_y) + 1)
        yout = yout * 10000
        print("出口浓度 yout:", yout)

        # 去除效率
        remove = (self.y_in - yout) / self.y_in
        print("去除效率 remove:", remove)
        return remove*100


# 使用类
# data_handler = DataHandler('H2S_latest.xlsx')
data_handler = DataHandler('H2S脱除.xlsx')
data_handler.load_data()
print(data_handler.data.head())
rpm = data_handler.data['rpm']
l_flow = data_handler.data['l_flow']
ph = data_handler.data['ph']
tannic_conc = data_handler.data['tannic_conc']
sodium_conc = data_handler.data['sodium_conc']
y_true = data_handler.data['y']
removal_efficiencies = []

# system = H2SHigeeSystem(rpm=rpm, l_flow=l_flow, ph=ph, tannic_conc=tannic_conc, sodium_conc=sodium_conc)
# removal_efficiencies = system.compute_removal_efficiency()
# data_handler.data['模型计算去除效率'] = removal_efficiencies

for i in range(len(rpm)):
    h2s_removal_system = H2SHigeeTheory(rpm=rpm[i], l_flow=l_flow[i], ph=ph[i], tannic_conc=tannic_conc[i], sodium_conc=sodium_conc[i])
    removal_efficiency = h2s_removal_system.compute_removal_efficiency()
    removal_efficiencies.append(removal_efficiency)

# 检查去除效率列表是否正确填充
print("Removal Efficiencies:", removal_efficiencies)
# 将结果保存到DataFrame
data_handler.data['模型计算去除效率'] = removal_efficiencies

# 保存计算结果到Excel文件
data_handler.data.to_excel('H2S脱除_模型计算结果.xlsx', index=False)
print("计算结果已保存到 H2S脱除_模型计算结果.xlsx 文件。")