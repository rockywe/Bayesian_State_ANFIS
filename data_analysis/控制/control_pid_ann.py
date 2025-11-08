# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from scipy.special import erf
import random
import warnings

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
Ns = 31
r1 = 42e-3
r2 = 146e-3
h = 20e-3
P = 101.325  # kPa
H = 709.1735  # Kpa*m3/mol

# Specific surface area of packing
a = 500  # m2/m3

# Volume of packing
v = np.pi * (r2**2 - r1**2) * (20e-3)

# Cross-sectional area
aa = np.pi * (1.6e-3)**2  # 16#

# Superoxygen concentration
c_oo2 = 7e-9  # mol/l

# Rate constant k_no-ooh
k_nooo2 = 6.9e9

# Apparent rate constant
k_appi = k_nooo2 * c_oo2

r = np.sqrt(r1 * r2)
D_no = 2.21e-9

gn2 = 2 / 3600
y_in = 0.0005

# PID Controller
class PIDController:
    def __init__(self, Kp, Ki, Kd, target):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target
        self.previous_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        error = self.target - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error

        return output

# Initial conditions
rpm = 400
l = 20e-3 / 3600  # l/s

# Target removal rate
target_remove = 0.985  # 目标脱除率
rpm_effection = 20  # 转速放大影响因子

# PID parameters (randomized within a range)
Kp = random.uniform(30, 70)
Ki = random.uniform(0.05, 0.2)
Kd = random.uniform(0.005, 0.02)
pid = PIDController(Kp, Ki, Kd, target_remove)

# Simulation parameters
dt = 1  # time step in seconds
num_steps = 1000  # Increase the number of simulation steps to generate more data

# Data storage for plotting
time_data = []
rpm_data = []
flow_data = []
remove_data = []
input_data = []
output_data = []

# Data preprocessing
scaler = StandardScaler()
output_scaler = StandardScaler()

# Train neural network placeholder
mlp = None

# Simulation loop
for step in range(num_steps):
    # Liquid flow rate
    L = l / aa  # m/s

    # Liquid film renewal time
    t_new = (r2 - r1) / (0.02107 * (L**0.2279) * ((r * rpm)**0.5448) * 31)

    # Liquid film mass transfer coefficient
    kl1 = (np.sqrt(D_no * k_appi) / t_new) * (t_new * erf(np.sqrt(k_appi * t_new)) +
          np.sqrt((t_new / np.pi) / k_appi) * np.exp(-k_appi * t_new) +
          (0.5 / k_appi) * erf(np.sqrt(k_appi * t_new)))

    ky_cal = 0.082 * 298.15 * 0.2700 * kl1
    a1 = (1 - y_in) / y_in
    a2 = a * np.pi * h * (r2**2 - r1**2) / gn2
    yno = 1 / (a1 * np.exp(a2 * ky_cal) + 1)

    remove = (y_in - yno) / y_in

    # PID control adjustment
    control_output = pid.update(remove, dt)

    # Adjust rpm and flow rate based on PID control output
    rpm = rpm + control_output * rpm_effection
    l += control_output * 1e-6  # small adjustment for flow rate

    # Collect training data
    error = pid.target - remove
    integral = pid.integral
    derivative = (error - pid.previous_error) / dt
    input_data.append([error, integral, derivative, rpm, l])  # Added RPM and flow rate as features
    output_data.append([pid.Kp, pid.Ki, pid.Kd])

    # Store data for plotting
    time_data.append(step * dt)
    rpm_data.append(rpm)
    flow_data.append(l)
    remove_data.append(remove)

    print(f"Step: {step}, RPM: {rpm:.2f}, Flow Rate: {l:.6f}, Removal: {remove:.4f}, Kp: {pid.Kp:.2f}, Ki: {pid.Ki:.4f}, Kd: {pid.Kd:.4f}")

    if abs(target_remove - remove) < 0.0001:  # Stop if close to target
        print("Target achieved")
        break

    # Update PID parameters dynamically after sufficient data is collected
    if step >= 10 and mlp is not None:
        current_state = [error, integral, derivative, rpm, l]
        current_state_normalized = scaler.transform([current_state])
        predicted_pid_params = mlp.predict(current_state_normalized)[0]
        predicted_pid_params = output_scaler.inverse_transform([predicted_pid_params])[0]  # Inverse transform to get original scale
        pid.Kp, pid.Ki, pid.Kd = predicted_pid_params

# Train the neural network after data collection
input_data_normalized = scaler.fit_transform(input_data)
output_data_normalized = output_scaler.fit_transform(output_data)
mlp = MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=5000, tol=1e-4)  # Increased complexity of the model
mlp.fit(input_data_normalized, output_data_normalized)

# Predict using the trained model
predictions_normalized = mlp.predict(input_data_normalized)
predictions = output_scaler.inverse_transform(predictions_normalized)

# Calculate R^2 score
r2 = r2_score(output_data, predictions)
print(f"R^2 Score of the Neural Network: {r2:.4f}")

# Plotting results
plt.figure(figsize=(14, 10))

# Plot RPM
plt.subplot(4, 1, 1)
plt.plot(time_data, rpm_data, label='RPM', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('RPM')
plt.title('RPM over Time')
plt.legend()

# Plot Flow Rate
plt.subplot(4, 1, 2)
plt.plot(time_data, flow_data, label='Flow Rate', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Flow Rate (l/s)')
plt.title('Flow Rate over Time')
plt.legend()

# Plot Removal Rate
plt.subplot(4, 1, 3)
plt.plot(time_data, remove_data, label='Removal Rate', color='red')
plt.axhline(target_remove, color='grey', linestyle='--', label='Target Removal Rate')
plt.xlabel('Time (s)')
plt.ylabel('Removal Rate')
plt.title('Removal Rate over Time')
plt.legend()

# Plot R^2 Score
# plt.subplot(4, 1, 4)
# plt.bar(['R^2 Score'], [r2], color='purple')
# plt.ylabel('R^2 Score')
# plt.title('Neural Network R^2 Score')
#
# plt.tight_layout()
# plt.show()
