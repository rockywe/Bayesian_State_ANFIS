import math
import matplotlib.pyplot as plt


def calculate_outlet_h2s(
        煤气进口流量_m3h,  # m³/h
        进口煤气温度_C,  # ℃
        进口煤气压力_kPa,  # kPa
        脱硫液流量_m3h,  # m³/h
        脱硫液温度_C,  # ℃
        脱硫液压力_kPa,  # kPa
        转速_RPM,  # RPM
        进口H2S浓度_ppm,  # ppm
        # 增强参数
        L_exponent=0.6,  # 脱硫液流量指数 (默认0.6)
        RPM_exponent=0.8,  # 转速影响指数 (默认0.8)
        G_exponent=-0.25,  # 煤气流量指数 (默认-0.25)
        gas_velocity_factor=1.2,  # 气体流速因子 (默认1.2)
        enhancement_factor=2.5,  # 强化因子 (默认2.5)
        contact_time_base=0.8  # 接触时间基准 (默认0.8)
):
    # ========================
    # 物理常数
    # ========================
    D_H2S = 1.8e-9  # H2S扩散系数 (m²/s)
    H_H2S = 483.0  # 亨利常数 (atm·m³/mol)
    alpha = 800  # 有效比表面积 (m²/m³)
    R_gas = 0.0821  # 气体常数 (L·atm/mol/K)
    liquid_density = 1100  # 脱硫液密度 (kg/m³)

    # ========================
    # 设备参数
    # ========================
    R_inner = 0.015  # 转子内径 (m)
    R_outer = 0.85  # 转子外径 (m)
    h_packing = 0.033  # 填料高度 (m)
    N_stages = 80  # 理论级数

    # ========================
    # 单位转换
    # ========================
    # 煤气参数
    G_m3s = 煤气进口流量_m3h / 3600
    T_gas = 进口煤气温度_C + 273.15
    P_total = 进口煤气压力_kPa / 101.325
    y_in = 进口H2S浓度_ppm * 1e-6

    # 脱硫液参数
    L_m3s = 脱硫液流量_m3h / 3600
    T_liquid = 脱硫液温度_C + 273.15

    # 转子参数
    R_avg = math.sqrt(R_inner * R_outer)
    omega = 转速_RPM * 2 * math.pi / 60

    # ========================
    # 增强传质模型
    # ========================
    def calculate_kL():
        # 增强离心力计算
        centrifugal_g = (omega ** 2 * R_avg) / 9.81
        # 指数增强计算
        kLa = 0.024 * enhancement_factor * (
                centrifugal_g ** (RPM_exponent * enhancement_factor) *
                L_m3s ** (L_exponent * enhancement_factor) *
                G_m3s ** (G_exponent * enhancement_factor)
        )
        return kLa / alpha

    # ========================
    # 动态物料平衡
    # ========================
    def mass_balance(kL):
        # 增强流速计算
        cross_area = math.pi * (R_outer ** 2 - R_inner ** 2)
        liquid_velocity = L_m3s / cross_area if cross_area != 0 else 0
        gas_velocity = G_m3s / cross_area if cross_area != 0 else 0

        # 增强接触时间
        combined_velocity = (liquid_velocity +
                             gas_velocity_factor * gas_velocity * enhancement_factor)
        residence_time = contact_time_base * h_packing / combined_velocity if combined_velocity != 0 else 0

        # 强化NTU计算
        NTU = kL * alpha * residence_time
        # 增强衰减模型
        y_out = y_in * math.exp(-NTU / (1 + NTU / (5 * enhancement_factor)))
        return y_out * 0.1  # 最终调整系数

    # ========================
    # 主计算流程
    # ========================
    try:
        kL = calculate_kL()
        y_out = mass_balance(kL)
        outlet_ppm = y_out * 1e6
        # 物理约束
        return max(0.0, min(outlet_ppm, 进口H2S浓度_ppm * 1.2))  # 允许20%超调
    except:
        return 0.0


# ========================
# 参数敏感性分析可视化
# ========================

def parameter_sensitivity_analysis():
    # 基准参数
    base_case = {
        "煤气进口流量_m3h": 800,
        "进口煤气温度_C": 40,
        "进口煤气压力_kPa": 20,
        "脱硫液流量_m3h": 20,
        "脱硫液温度_C": 35,
        "脱硫液压力_kPa": 40,
        "转速_RPM": 1200,
        "进口H2S浓度_ppm": 1000
    }

    # 分析参数配置
    analysis_config = [
        {
            "title": "脱硫液流量敏感性分析",
            "param": "脱硫液流量_m3h",
            "range": (0, 100),
            "step": 10,  # 新增步长参数
            "xlabel": "脱硫液流量 (m³/h)",
            "enhance_params": {"L_exponent": 0.8, "enhancement_factor": 3.0},
            "color": "#2ecc71"
        },
        {
            "title": "转速敏感性分析",
            "param": "转速_RPM",
            "range": (800, 2000),
            "step": 50,
            "xlabel": "转速 (RPM)",
            "enhance_params": {"RPM_exponent": 1.0, "enhancement_factor": 3.0},
            "color": "#e74c3c"
        },
        {
            "title": "煤气流量敏感性分析",
            "param": "煤气进口流量_m3h",
            "range": (500, 1500),
            "step": 50,
            "xlabel": "煤气流量 (m³/h)",
            "enhance_params": {"G_exponent": -0.4, "gas_velocity_factor": 1.5},
            "color": "#3498db"
        }
    ]

    # 全局样式设置
    plt.rcParams.update({
        'font.sans-serif': 'SimHei',  # 中文显示
        'axes.unicode_minus': False,
        'figure.dpi': 120,  # 图像清晰度
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4
    })

    # 生成独立图表
    for config in analysis_config:
        # 创建独立画布
        plt.figure(figsize=(8, 5), facecolor='#f5f6fa')

        # 数据准备
        x_values = []
        y_normal = []
        y_enhanced = []

        # 参数扫描
        for value in range(config["range"][0], config["range"][1] + config["step"], config["step"]):
            # 复制基准参数
            current_params = base_case.copy()
            current_params[config["param"]] = value

            # 计算默认参数结果
            normal_result = calculate_outlet_h2s(**current_params)

            # 计算增强参数结果
            enhanced_params = current_params.copy()
            enhanced_params.update(config["enhance_params"])
            enhanced_result = calculate_outlet_h2s(**enhanced_params)

            # 记录数据
            x_values.append(value)
            y_normal.append(normal_result)
            y_enhanced.append(enhanced_result)

        # 绘制主曲线
        plt.plot(x_values, y_normal,
                 color=config["color"],
                 linestyle='-',
                 linewidth=2.5,
                 marker='o',
                 markersize=8,
                 markerfacecolor='white',
                 markeredgewidth=1.5,
                 label='默认参数')

        plt.plot(x_values, y_enhanced,
                 color=config["color"],
                 linestyle='--',
                 linewidth=2.5,
                 marker='s',
                 markersize=8,
                 markerfacecolor='white',
                 markeredgewidth=1.5,
                 label='参数强化')

        # 图表装饰
        plt.title(config["title"], fontsize=14, pad=20)
        plt.xlabel(config["xlabel"], fontsize=12, labelpad=10)
        plt.ylabel("出口H₂S浓度 (ppm)", fontsize=12, labelpad=10)

        # 坐标轴优化
        plt.xlim(min(x_values) - 10, max(x_values) + 10)
        plt.ylim(0, max(max(y_normal), max(y_enhanced)) * 1.2)

        # 图例设置
        plt.legend(
            loc='upper right',
            frameon=False,
            fontsize=10,
            handlelength=2.5
        )

        # 显示图表
        plt.tight_layout(pad=2.0)
        plt.show()




# ========================
# 执行程序
# ========================
if __name__ == "__main__":
    # 单点测试
    test_case = {
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

    print("【增强测试】出口浓度:", calculate_outlet_h2s(**test_case))

    # 执行可视化分析
    parameter_sensitivity_analysis()