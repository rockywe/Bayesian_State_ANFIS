import math


def calculate_outlet_h2s(
        煤气进口流量_m3h,  # m³/h
        进口煤气温度_C,  # ℃
        进口煤气压力_kPa,  # kPa
        脱硫液流量_m3h,  # m³/h
        脱硫液温度_C,  # ℃
        脱硫液压力_kPa,  # kPa
        转速_RPM,  # RPM
        进口H2S浓度_ppm,  # ppm
        L_exponent=0.18,  # 脱硫液流量指数修正 (默认原值0.18)
        RPM_exponent=0.33,  # 转速在离心力中的指数修正 (默认原值0.33)
        G_exponent=-0.05,  # 煤气流量指数修正 (默认原值-0.05)
        gas_velocity_factor=0.0  # 气体流速对接触时间的影响因子 (默认0不启用)
):
    # ----------------------------
    # 常量参数 (根据实际工艺调整)
    # ----------------------------
    D_H2S = 1.8e-9  # H2S扩散系数 (m²/s)
    H_H2S = 483.0  # 亨利常数 (atm·m³/mol)
    alpha = 800  # 有效比表面积 (m²/m³)
    R_gas = 0.0821  # 理想气体常数 (L·atm/mol/K)
    liquid_density = 1100  # 脱硫液密度 (kg/m³)

    # ----------------------------
    # 设备几何参数 (根据实际设备调整)
    # ----------------------------
    R_inner = 0.015  # 转子内径 (m)
    R_outer = 0.85  # 转子外径 (m)
    h_packing = 0.033  # 填料层高度 (m)
    N_stages = 80  # 理论分离级数

    # ----------------------------
    # 单位转换
    # ----------------------------
    # 气体参数
    G_m3s = 煤气进口流量_m3h / 3600  # m³/h → m³/s
    T_gas = 进口煤气温度_C + 273.15  # ℃ → K
    P_total = 进口煤气压力_kPa / 101.325  # kPa → atm
    y_in = 进口H2S浓度_ppm * 1e-6  # ppm → 摩尔分数

    # 液体参数
    L_m3s = 脱硫液流量_m3h / 3600  # m³/h → m³/s
    T_liquid = 脱硫液温度_C + 273.15  # ℃ → K

    # 转子参数
    R_avg = math.sqrt(R_inner * R_outer)  # 平均半径 (m)
    omega = 转速_RPM * 2 * math.pi / 60  # RPM → rad/s

    # ----------------------------
    # 核心计算模型 (带修正参数)
    # ----------------------------
    # 1. 体积传质系数计算 (修正指数)
    def calculate_kL():
        centrifugal_g = (omega ** 2 * R_avg) / 9.81  # 相对离心力(RCF)
        # 使用修正后的指数
        kLa = 0.024 * (centrifugal_g ** RPM_exponent) * (L_m3s ** L_exponent) * (G_m3s ** G_exponent)
        return kLa / alpha  # 转换为kL (m/s)

    # 2. 气液平衡计算
    def equilibrium_concentration():
        C_star = y_in * P_total / H_H2S  # 气相平衡浓度 (mol/m³)
        return C_star

    # 3. 物料平衡方程 (添加气体流速影响)
    def mass_balance(kL):
        # 计算液体和气体流速
        cross_area = math.pi * (R_outer ** 2 - R_inner ** 2)  # 填料横截面积 (m²)
        liquid_velocity = L_m3s / cross_area if cross_area != 0 else 0  # 液体流速 (m/s)
        gas_velocity = G_m3s / cross_area if cross_area != 0 else 0  # 气体流速 (m/s)

        # 综合流速计算 (考虑气体影响因子)
        combined_velocity = liquid_velocity + gas_velocity_factor * gas_velocity
        residence_time = h_packing / combined_velocity if combined_velocity != 0 else 0

        # 传质单元数
        NTU = kL * alpha * residence_time

        # 出口浓度计算 (指数衰减模型)
        y_out = y_in * math.exp(-NTU / (1 + NTU))
        y_out = y_out / 10  # 经验调整系数
        return y_out

    # ----------------------------
    # 主计算流程
    # ----------------------------
    kL = calculate_kL()
    y_out = mass_balance(kL)

    # 结果转换与合理性约束
    outlet_ppm = y_out * 1e6  # 转回ppm
    return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm))


# ----------------------------
# 示例测试 (带修正参数)
# ----------------------------
if __name__ == "__main__":
    # 正常工况测试
    输入参数 = {
        "煤气进口流量_m3h": 796,
        "进口煤气温度_C": 43,
        "进口煤气压力_kPa": 18,
        "脱硫液流量_m3h": 18,
        "脱硫液温度_C": 38,
        "脱硫液压力_kPa": 41,
        "转速_RPM": 1200,
        "进口H2S浓度_ppm": 900,
        "L_exponent": 0.3,  # 加强脱硫液影响
        "RPM_exponent": 0.5,  # 加强转速影响
        "G_exponent": -0.2,  # 加强煤气流量影响
        "gas_velocity_factor": 0.5  # 启用气体流速影响
    }

    出口浓度 = calculate_outlet_h2s(**输入参数)
    print(f"【修正后正常工况】出口H2S浓度: {出口浓度:.1f} ppm")

    # 敏感性测试 - 提高进口浓度
    高浓度输入 = 输入参数.copy()
    高浓度输入["进口H2S浓度_ppm"] = 1500
    print(f"【高浓度输入】出口浓度: {calculate_outlet_h2s(**高浓度输入):.1f} ppm")

    # 敏感性测试 - 提高转速
    高速输入 = 输入参数.copy()
    高速输入["转速_RPM"] = 1800
    print(f"【高转速运行】出口浓度: {calculate_outlet_h2s(**高速输入):.1f} ppm")

    # 敏感性测试 - 增加液体流量
    高液量输入 = 输入参数.copy()
    高液量输入["脱硫液流量_m3h"] = 36
    print(f"【高脱硫液量】出口浓度: {calculate_outlet_h2s(**高液量输入):.1f} ppm")