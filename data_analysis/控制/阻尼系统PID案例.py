import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 系统参数
m = 1.0  # 质量
b = 0.2  # 阻尼系数
k = 1.0  # 弹簧刚度

# PID参数
Kp = 10.0
Ki = 1.0
Kd = 1.0

# 目标值
setpoint = 1.0

# 系统动力学模型
def model(y, t, u):
    x, v = y
    dxdt = v
    dvdt = (u - b*v - k*x) / m
    return [dxdt, dvdt]

# PID控制器
def pid_control(y, t, setpoint, Kp, Ki, Kd, integral, previous_error):
    error = setpoint - y[0]
    integral += error * t
    derivative = (error - previous_error) / t
    u = Kp * error + Ki * integral + Kd * derivative
    previous_error = error
    return u, integral, previous_error

# 仿真参数
t = np.linspace(0, 10, 100)
y0 = [0, 0]
integral = 0
previous_error = 0
y = np.zeros((len(t), 2))
u = np.zeros(len(t))

# 仿真
for i in range(len(t)-1):
    u[i], integral, previous_error = pid_control(y[i], t[i+1] - t[i], setpoint, Kp, Ki, Kd, integral, previous_error)
    y[i+1] = odeint(model, y[i], [t[i], t[i+1]], args=(u[i],))[1]

# 绘图
plt.plot(t, y[:, 0], label='Position')
plt.plot(t, setpoint*np.ones_like(t), '--', label='Setpoint')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.show()
