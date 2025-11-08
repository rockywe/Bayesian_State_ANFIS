# -----------------------------
# 导入库
# -----------------------------
import numpy as np
from scipy.interpolate import interp1d
from math import pi, sqrt, exp, log
from fipy import Grid1D, CellVariable, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy import viewers


# -----------------------------
# 参数类
# -----------------------------
class ReactorParameters:
    def __init__(self):
        # 实验条件
        self.pH = 12.5
        self.ch = 1.0  # H2O2浓度 (mol/m³)
        self.T = 298.15  # 温度 (K)
        self.Pno = 101.325  # NO分压 (kPa)
        self.Hno = 709.1725  # 亨利系数 (kPa·m³/mol)
        self.ooh = self._calculate_ooh()  # HOO⁻浓度

        # RPB参数
        self.rpm = 1200
        self.r1 = 26e-3  # 内径 (m)
        self.r2 = 70e-3  # 外径 (m)
        self.h = 25e-3  # 高度 (m)
        self.ap = 550  # 比表面积 (m²/m³)
        self.r = sqrt((self.r1 ** 2 + self.r2 ** 2) / 2)  # 平均半径

        # 扩散系数插值数据
        self.rpm_data = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])
        self.De_data = np.array([5.3064, 6.2257, 6.7727, 7.1618, 7.4629, 7.7081, 7.9144, 8.0818]) * 1e-10

    def _calculate_ooh(self) -> float:
        """计算 HOO⁻ 浓度"""
        return 1.0 / (1.0 + 10 ** (11.73 - self.pH)) * self.ch


# -----------------------------
# 计算液膜参数
# -----------------------------
def calculate_reactor_params(params: ReactorParameters) -> dict:
    """计算液膜特性与传质参数"""
    omega = 2 * pi * params.rpm / 60.0  # 角速度 (rad/s)
    De = interp1d(params.rpm_data, params.De_data, kind='linear')(params.rpm)

    # 液滴径向速度 (m/s)
    u0 = 0.02107 * (20e-3 / 3600) ** 0.2279 * (omega ** 2 * params.r)

    # 液膜更新时间 (s)
    ts = (params.r2 - params.r1) / (u0 * 31.0)

    # 液膜厚度 (m)
    delta = 2.4e-5 * (params.r * omega ** 2) ** (-0.62) * u0 ** 0.47

    return {
        'De': De,
        'u0': u0,
        'ts': ts,
        'delta': delta,
    }


# -----------------------------
# 导入库
# -----------------------------
import numpy as np
from scipy.interpolate import interp1d
from math import pi, sqrt
from fipy import Grid1D, CellVariable, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy import viewers


# -----------------------------
# 参数类(完整物理单位处理)
# -----------------------------
class ReactorParameters:
    def __init__(self):
        # 实验条件
        self.pH = 12.5
        self.ch = 1.0  # H2O2浓度 (mol/m³)
        self.T = 298.15  # 温度 (K)
        self.Pno = 101.325  # NO分压 (kPa)
        self.Hno = 709.1725  # 亨利系数 (kPa·m³/mol)
        self.ooh = self._calculate_ooh()  # HOO⁻浓度

        # RPB参数
        self.rpm = 1200
        self.r1 = 26e-3  # 内径 (m)
        self.r2 = 70e-3  # 外径 (m)
        self.h = 25e-3  # 高度 (m)
        self.ap = 550  # 比表面积 (m²/m³)
        self.r = sqrt((self.r1 ** 2 + self.r2 ** 2) / 2)  # 平均半径

        # 扩散系数插值数据 (转换为m²/s)
        self.rpm_data = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])
        self.De_data = np.array([5.3064, 6.2257, 6.7727, 7.1618, 7.4629, 7.7081, 7.9144, 8.0818]) * 1e-10

    def _calculate_ooh(self) -> float:
        """计算 HOO⁻ 浓度"""
        return float(1.0 / (1.0 + 10 ** (11.73 - self.pH)) * self.ch)


# -----------------------------
# 液膜参数计算(增加数值安全处理)
# -----------------------------
def calculate_reactor_params(params: ReactorParameters) -> dict:
    """计算液膜特性与传质参数"""
    try:
        omega = 2 * pi * params.rpm / 60.0
        De = float(interp1d(params.rpm_data, params.De_data)(params.rpm))

        # 液滴径向速度 (m/s)
        u0 = 0.02107 * (20e-3 / 3600) ** 0.2279 * (omega ** 2 * params.r)
        if u0 <= 0:
            raise ValueError("液滴速度计算异常，请检查参数输入")

        # 液膜更新时间 (s)
        ts = float((params.r2 - params.r1) / (u0 * 31.0))

        # 液膜厚度 (m) 使用经验公式
        delta = 2.4e-5 * (params.r * omega ** 2) ** (-0.62) * u0 ** 0.47

        return {
            'De': De,
            'u0': float(u0),
            'ts': ts,
            'delta': float(delta),
        }
    except Exception as e:
        raise RuntimeError(f"液膜参数计算失败: {str(e)}")


# -----------------------------
# PDE求解模块(完整数值处理)
# -----------------------------
def solve_pde(params: ReactorParameters, reactor_params: dict):
    """求解耦合扩散-反应方程"""
    # 参数提取与类型安全
    delta = float(reactor_params['delta'])
    De = float(reactor_params['De'])
    ts = float(reactor_params['ts'])

    # 扩散系数修正
    D1 = float(2.21e-9 * De * (params.rpm / 1000) ** 0.5)
    D2 = float(1.97e-9 * De * (params.rpm / 1000) ** 0.5)
    k = 103.4  # 反应速率常数 (m³/(mol·s))

    # 网格系统
    nx = 100
    mesh = Grid1D(dx=delta / nx, nx=nx)

    # 定义浓度变量
    c_NO = CellVariable(name="NO", mesh=mesh, value=0.0, hasOld=True)
    c_HOO = CellVariable(name="HOO⁻", mesh=mesh, value=0.0, hasOld=True)

    # 边界条件(显式类型转换)
    C_in = float(params.Pno / params.Hno)
    c_NO.constrain(C_in, mesh.facesLeft)
    c_HOO.constrain(float(params.ooh), mesh.facesLeft)

    # 零梯度边界条件
    c_NO.faceGrad.constrain(0, mesh.facesRight)
    c_HOO.faceGrad.constrain(0, mesh.facesRight)

    # 定义控制方程
    eq_NO = TransientTerm(var=c_NO) == DiffusionTerm(coeff=D1, var=c_NO) - ImplicitSourceTerm(coeff=k * c_HOO, var=c_NO)
    eq_HOO = TransientTerm(var=c_HOO) == DiffusionTerm(coeff=D2, var=c_HOO) - ImplicitSourceTerm(coeff=k * c_NO,
                                                                                                 var=c_HOO)
    coupled_eq = eq_NO & eq_HOO

    # 时间步进参数
    dt = float(0.25 * delta ** 2 / max(D1, D2))
    steps = int(ts / dt) + 1

    # 可视化设置(修复警告)
    viewer = viewers.MatplotlibViewer(
        vars=(c_NO, c_HOO),
        datamin=0.0,
        datamax=C_in * 1.1,
        title="浓度分布演化"
    )

    # 时间迭代求解
    for step in range(steps):
        c_NO.updateOld()
        c_HOO.updateOld()

        residual = 1e5
        while residual > 1e-5:
            residual = coupled_eq.sweep(dt=dt)

        if step % 10 == 0:
            viewer.plot()

    # 计算去除率
    C_out = float(c_NO.faceValue[mesh.facesRight.value][0])
    removal = float(1 - C_out / C_in)

    return removal


# -----------------------------
# 主程序(完整异常处理)
# -----------------------------
if __name__ == "__main__":
    try:
        params = ReactorParameters()
        reactor_params = calculate_reactor_params(params)

        print("计算参数摘要:")
        print(f"- 液膜厚度: {reactor_params['delta'] * 1e6:.2f} μm")
        print(f"- 扩散系数(D1): {reactor_params['De'] * 1e10:.2f} ×10⁻¹⁰ m²/s")
        print(f"- 特征时间: {reactor_params['ts']:.3f} s")

        removal = solve_pde(params, reactor_params)

        print("\n" + "=" * 40)
        print(f"模拟结果: NO去除率 = {removal * 100:.2f}%")
        print("=" * 40)

    except Exception as e:
        print("\n" + "!" * 40)
        print(f"计算终止，发生错误: {str(e)}")
        print("!" * 40)