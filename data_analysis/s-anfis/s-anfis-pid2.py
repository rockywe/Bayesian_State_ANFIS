import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
import matplotlib.colors as mcolors
import copy
import os
from datetime import datetime
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sanfis import SANFIS
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica World', 'Arial', 'Arial Unicode MS']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['path.simplify'] = True
plt.rcParams['path.snap'] = True

# 创建保存目录
save_dir = 'sanfis_process_optimization'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pic_dir = './pic'
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# [数据加载和预处理代码保持不变...]
file_path = './data/脱硫数据整理2.xlsx'
try:
    data_df = pd.read_excel(file_path, sheet_name='Sheet1')
except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径：{file_path}")
    exit()

# 数据预处理
column_rename_dict = {
    '煤气进口流量': 'Gas_Inlet_Flow',
    '进口煤气温度': 'Gas_Inlet_Temperature',
    '进口煤气压力': 'Gas_Inlet_Pressure',
    '脱硫液流量': 'Desulfurization_Liquid_Flow',
    '脱硫液温度': 'Desulfurization_Liquid_Temperature',
    '脱硫液压力': 'Desulfurization_Liquid_Pressure',
    '转速': 'Rotation_Speed',
    '进口H2S浓度': 'H2S_Inlet_Concentration',
    '出口H2S浓度': 'H2S_Outlet_Concentration'
}

data_df.rename(columns=column_rename_dict, inplace=True)

input_features = [
    'Gas_Inlet_Flow',  # 0 - 煤气进口流量
    'Desulfurization_Liquid_Flow',  # 1 - 脱硫液流量
    'Rotation_Speed',  # 2 - 转速
    'H2S_Inlet_Concentration'  # 3 - 进口H2S浓度
]
output_feature = 'H2S_Outlet_Concentration'

# 数据清理和标准化
data_clean = data_df[input_features + [output_feature]].dropna()
X = data_clean[input_features].values
y = data_clean[output_feature].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

S_train_tensor = X_train_tensor
S_test_tensor = X_test_tensor

train_data = [S_train_tensor, X_train_tensor, y_train_tensor]
valid_data = [S_test_tensor, X_test_tensor, y_test_tensor]

# 修正的参数范围
X_original_ranges = {
    'Gas_Inlet_Flow': (500.00, 1042.52),
    'Desulfurization_Liquid_Flow': (11.50, 22.28),
    'Rotation_Speed': (40.00, 70.00),
    'H2S_Inlet_Concentration': (600.00, 1500.00)
}


def train_sanfis_model():
    """训练SANFIS模型"""
    print(f"\n{'=' * 60}")
    print(f"训练SANFIS模型")
    print(f"{'=' * 60}")

    # 模型配置
    n_input_features = X_scaled.shape[1]
    n_memb_funcs_per_input = 3

    membfuncs_config = []
    for i in range(n_input_features):
        mu_values = np.linspace(0.1, 0.9, n_memb_funcs_per_input).tolist()
        sigma_values = [0.2] * n_memb_funcs_per_input

        membfuncs_config.append({
            'function': 'gaussian',
            'n_memb': n_memb_funcs_per_input,
            'params': {
                'mu': {'value': mu_values, 'trainable': True},
                'sigma': {'value': sigma_values, 'trainable': True}
            }
        })

    # 初始化模型
    model = SANFIS(
        membfuncs=membfuncs_config,
        n_input=n_input_features,
        to_device='cpu',
        scale='Std'
    )

    # 训练模型
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss(reduction='mean')

    epochs = 2000
    print_interval = 100

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        train_pred = model(train_data[0], train_data[1])
        train_loss = loss_function(train_pred, train_data[2])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % print_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_pred = model(valid_data[0], valid_data[1])
                valid_loss = loss_function(valid_pred, valid_data[2])
            model.train()

            print(f"Epoch [{epoch + 1}/{epochs}] - "
                  f"Train Loss: {train_loss.item():.6f}, "
                  f"Valid Loss: {valid_loss.item():.6f}")

    # 最终评估
    model.eval()
    with torch.no_grad():
        y_pred_scaled_tensor = model(S_test_tensor, X_test_tensor)

    y_pred_scaled = y_pred_scaled_tensor.detach().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_np)

    mse = mean_squared_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    print(f"\n模型训练完成:")
    print(f"MSE: {mse:.6f}")
    print(f"R²: {r2:.6f}")

    return model, mse, r2, membfuncs_config


def save_sanfis_model(model, membfuncs_config, mse, r2):
    """保存训练好的SANFIS模型"""
    model_save_path = os.path.join(save_dir, f'trained_sanfis_model_{timestamp}.pth')

    # 保存完整的模型信息
    torch.save({
        'model_state_dict': model.state_dict(),
        'membfuncs_config': membfuncs_config,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'input_features': input_features,
        'output_feature': output_feature,
        'model_performance': {'mse': mse, 'r2': r2},
        'parameter_ranges': X_original_ranges,
        'n_input_features': len(input_features),
        'timestamp': timestamp
    }, model_save_path)

    print(f"\n训练好的模型已保存: {model_save_path}")
    return model_save_path


def load_sanfis_model(model_path):
    """加载训练好的SANFIS模型"""
    print(f"加载模型: {model_path}")

    # 加载模型数据
    checkpoint = torch.load(model_path, map_location='cpu')

    # 重建模型
    model = SANFIS(
        membfuncs=checkpoint['membfuncs_config'],
        n_input=checkpoint['n_input_features'],
        to_device='cpu',
        scale='Std'
    )

    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载成功!")
    print(f"模型性能 - MSE: {checkpoint['model_performance']['mse']:.6f}, "
          f"R²: {checkpoint['model_performance']['r2']:.6f}")

    return model, checkpoint


class TraditionalPIDController:
    """传统PID控制器（用于对比）"""

    def __init__(self, kp=0.5, ki=0.02, kd=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # PID状态变量
        self.prev_error = 0.0
        self.integral = 0.0
        self.dt = 1.0

        # 控制变量范围
        self.control_ranges = {
            'rotation_speed': (40.0, 70.0),
            'gas_flow': (500.0, 1042.52),
            'liquid_flow': (11.5, 22.28)
        }

        print(f"传统PID控制器初始化完成 (Kp={kp}, Ki={ki}, Kd={kd})")

    def traditional_pid_control(self, setpoint, current_output, current_gas_flow,
                                current_liquid_flow, current_rotation_speed):
        """传统PID控制"""
        # 计算误差
        error = setpoint - current_output

        # PID计算
        proportional = self.kp * error

        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -5.0, 5.0)  # 积分限幅
        integral_term = self.ki * self.integral

        derivative = (error - self.prev_error) / self.dt
        derivative_term = self.kd * derivative

        pid_output = proportional + integral_term + derivative_term
        pid_output = np.clip(pid_output, -5.0, 5.0)  # 输出限幅

        # 控制变量调整（更保守的调整）
        speed_adjustment = pid_output * 0.1
        new_rotation_speed = current_rotation_speed + speed_adjustment
        new_rotation_speed = np.clip(new_rotation_speed,
                                     self.control_ranges['rotation_speed'][0],
                                     self.control_ranges['rotation_speed'][1])

        gas_adjustment = pid_output * 0.5
        new_gas_flow = current_gas_flow + gas_adjustment
        new_gas_flow = np.clip(new_gas_flow,
                               self.control_ranges['gas_flow'][0],
                               self.control_ranges['gas_flow'][1])

        liquid_adjustment = -pid_output * 0.05
        new_liquid_flow = current_liquid_flow + liquid_adjustment
        new_liquid_flow = np.clip(new_liquid_flow,
                                  self.control_ranges['liquid_flow'][0],
                                  self.control_ranges['liquid_flow'][1])

        self.prev_error = error

        return {
            'rotation_speed': new_rotation_speed,
            'gas_flow': new_gas_flow,
            'liquid_flow': new_liquid_flow,
            'error': error,
            'pid_output': pid_output
        }

    def reset_pid(self):
        self.prev_error = 0.0
        self.integral = 0.0


class FuzzySupervisedPIDController:
    """基于S-ANFIS的模糊监督PID控制器（优化版）"""

    def __init__(self, sanfis_model, scaler_X, scaler_y, input_features):
        self.sanfis_model = sanfis_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.input_features = input_features

        # 更保守的PID参数
        self.kp = 0.3  # 进一步降低
        self.ki = 0.01  # 进一步降低
        self.kd = 0.005  # 进一步降低

        # PID状态变量
        self.prev_error = 0.0
        self.integral = 0.0
        self.dt = 1.0

        # 控制变量范围
        self.control_ranges = {
            'rotation_speed': (40.0, 70.0),
            'gas_flow': (500.0, 1042.52),
            'liquid_flow': (11.5, 22.28)
        }

        # 输出范围检查
        self.output_min = 1.0

        print("模糊监督PID控制器初始化完成（优化版）")

    def predict_h2s_output(self, gas_flow, liquid_flow, rotation_speed, inlet_h2s):
        """使用S-ANFIS预测H2S出口浓度，确保结果大于1"""
        # 确保输入参数在合理范围内
        gas_flow = np.clip(gas_flow, self.control_ranges['gas_flow'][0],
                           self.control_ranges['gas_flow'][1])
        liquid_flow = np.clip(liquid_flow, self.control_ranges['liquid_flow'][0],
                              self.control_ranges['liquid_flow'][1])
        rotation_speed = np.clip(rotation_speed, self.control_ranges['rotation_speed'][0],
                                 self.control_ranges['rotation_speed'][1])
        inlet_h2s = np.clip(inlet_h2s, 600.0, 1500.0)

        # 构建输入向量
        input_vector = np.array([[gas_flow, liquid_flow, rotation_speed, inlet_h2s]])

        # 标准化输入
        input_scaled = self.scaler_X.transform(input_vector)

        # 转换为张量
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # 预测
        with torch.no_grad():
            output_scaled = self.sanfis_model(input_tensor, input_tensor)
            output_original = self.scaler_y.inverse_transform(output_scaled.numpy())

        # 确保输出大于最小值
        predicted_output = max(output_original[0, 0], self.output_min)

        return predicted_output

    def fuzzy_pid_control(self, setpoint, current_gas_flow, current_liquid_flow,
                          current_rotation_speed, inlet_h2s):
        """模糊监督PID控制（优化版）"""

        # 1. 预测当前输出
        predicted_output = self.predict_h2s_output(
            current_gas_flow, current_liquid_flow, current_rotation_speed, inlet_h2s
        )

        # 2. 计算误差
        error = setpoint - predicted_output

        # 3. PID计算
        proportional = self.kp * error

        # 积分项（更严格的限幅）
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -3.0, 3.0)  # 更小的积分限幅
        integral_term = self.ki * self.integral

        # 微分项
        derivative = (error - self.prev_error) / self.dt
        derivative_term = self.kd * derivative

        # PID输出
        pid_output = proportional + integral_term + derivative_term

        # 4. 模糊监督调整（更保守）
        error_abs = abs(error)

        if error_abs > 3.0:  # 大误差
            kp_adj = 1.1  # 降低调整幅度
            ki_adj = 0.5
            kd_adj = 1.0
        elif error_abs > 1.0:  # 中等误差
            kp_adj = 1.0
            ki_adj = 1.0
            kd_adj = 1.0
        else:  # 小误差
            kp_adj = 0.9
            ki_adj = 1.1
            kd_adj = 0.8

        adjusted_pid_output = (proportional * kp_adj +
                               integral_term * ki_adj +
                               derivative_term * kd_adj)

        # 更严格的输出限制
        adjusted_pid_output = np.clip(adjusted_pid_output, -2.0, 2.0)

        # 5. 极小的控制变量调整
        # 转速调整
        speed_adjustment = adjusted_pid_output * 0.05  # 更小的调整幅度
        new_rotation_speed = current_rotation_speed + speed_adjustment
        new_rotation_speed = np.clip(new_rotation_speed,
                                     self.control_ranges['rotation_speed'][0],
                                     self.control_ranges['rotation_speed'][1])

        # 进气量调整
        gas_adjustment = adjusted_pid_output * 0.2  # 更小的调整幅度
        new_gas_flow = current_gas_flow + gas_adjustment
        new_gas_flow = np.clip(new_gas_flow,
                               self.control_ranges['gas_flow'][0],
                               self.control_ranges['gas_flow'][1])

        # 脱硫液量调整
        liquid_adjustment = -adjusted_pid_output * 0.02  # 更小的调整幅度
        new_liquid_flow = current_liquid_flow + liquid_adjustment
        new_liquid_flow = np.clip(new_liquid_flow,
                                  self.control_ranges['liquid_flow'][0],
                                  self.control_ranges['liquid_flow'][1])

        # 预测新的输出
        new_predicted_output = self.predict_h2s_output(
            new_gas_flow, new_liquid_flow, new_rotation_speed, inlet_h2s
        )

        self.prev_error = error

        return {
            'rotation_speed': new_rotation_speed,
            'gas_flow': new_gas_flow,
            'liquid_flow': new_liquid_flow,
            'predicted_output': new_predicted_output,
            'current_output': predicted_output,
            'error': error,
            'pid_output': adjusted_pid_output
        }

    def reset_pid(self):
        self.prev_error = 0.0
        self.integral = 0.0


def simulate_control_comparison(fuzzy_controller, traditional_controller, sanfis_model,
                                scaler_X, scaler_y, setpoint, initial_conditions,
                                inlet_h2s_profile, simulation_steps=50):
    """对比模糊PID和传统PID控制"""

    print(f"\n开始PID控制对比仿真")
    print(f"目标H2S出口浓度: {setpoint:.2f} mg/m³")
    print(f"仿真步数: {simulation_steps}")

    # 初始化数据记录
    results = {
        'fuzzy_pid': {
            'time_steps': [], 'gas_flows': [], 'liquid_flows': [], 'rotation_speeds': [],
            'predicted_outputs': [], 'errors': [], 'pid_outputs': []
        },
        'traditional_pid': {
            'time_steps': [], 'gas_flows': [], 'liquid_flows': [], 'rotation_speeds': [],
            'predicted_outputs': [], 'errors': [], 'pid_outputs': []
        }
    }

    # 模糊PID仿真
    print("\n=== 模糊监督PID控制仿真 ===")
    fuzzy_controller.reset_pid()
    current_gas_flow = initial_conditions['gas_flow']
    current_liquid_flow = initial_conditions['liquid_flow']
    current_rotation_speed = initial_conditions['rotation_speed']

    for step in range(simulation_steps):
        inlet_h2s = inlet_h2s_profile[step] if len(inlet_h2s_profile) > step else inlet_h2s_profile[-1]

        control_result = fuzzy_controller.fuzzy_pid_control(
            setpoint, current_gas_flow, current_liquid_flow,
            current_rotation_speed, inlet_h2s
        )

        current_gas_flow = control_result['gas_flow']
        current_liquid_flow = control_result['liquid_flow']
        current_rotation_speed = control_result['rotation_speed']

        results['fuzzy_pid']['time_steps'].append(step)
        results['fuzzy_pid']['gas_flows'].append(current_gas_flow)
        results['fuzzy_pid']['liquid_flows'].append(current_liquid_flow)
        results['fuzzy_pid']['rotation_speeds'].append(current_rotation_speed)
        results['fuzzy_pid']['predicted_outputs'].append(control_result['predicted_output'])
        results['fuzzy_pid']['errors'].append(control_result['error'])
        results['fuzzy_pid']['pid_outputs'].append(control_result['pid_output'])

    # 传统PID仿真
    print("\n=== 传统PID控制仿真 ===")
    traditional_controller.reset_pid()
    current_gas_flow = initial_conditions['gas_flow']
    current_liquid_flow = initial_conditions['liquid_flow']
    current_rotation_speed = initial_conditions['rotation_speed']

    for step in range(simulation_steps):
        inlet_h2s = inlet_h2s_profile[step] if len(inlet_h2s_profile) > step else inlet_h2s_profile[-1]

        # 使用S-ANFIS预测当前输出
        current_output = fuzzy_controller.predict_h2s_output(
            current_gas_flow, current_liquid_flow, current_rotation_speed, inlet_h2s
        )

        control_result = traditional_controller.traditional_pid_control(
            setpoint, current_output, current_gas_flow, current_liquid_flow, current_rotation_speed
        )

        current_gas_flow = control_result['gas_flow']
        current_liquid_flow = control_result['liquid_flow']
        current_rotation_speed = control_result['rotation_speed']

        # 预测新的输出
        new_output = fuzzy_controller.predict_h2s_output(
            current_gas_flow, current_liquid_flow, current_rotation_speed, inlet_h2s
        )

        results['traditional_pid']['time_steps'].append(step)
        results['traditional_pid']['gas_flows'].append(current_gas_flow)
        results['traditional_pid']['liquid_flows'].append(current_liquid_flow)
        results['traditional_pid']['rotation_speeds'].append(current_rotation_speed)
        results['traditional_pid']['predicted_outputs'].append(new_output)
        results['traditional_pid']['errors'].append(control_result['error'])
        results['traditional_pid']['pid_outputs'].append(control_result['pid_output'])

    # 转换为numpy数组
    for controller_type in results:
        for key in results[controller_type]:
            results[controller_type][key] = np.array(results[controller_type][key])

    results['setpoint'] = setpoint
    results['inlet_h2s_profile'] = inlet_h2s_profile

    return results


# ==================== MODIFICATION START ====================
# This is the updated plotting function with a clean, professional style.
def plot_control_comparison(comparison_results):
    """绘制PID控制对比结果，采用简洁、专业的出版风格"""

    results = comparison_results

    # 创建一个 2x2 的网格图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)

    # 定义统一的线条宽度和刻度标签字号
    line_width = 2.0
    tick_label_size = 16  # XY轴刻度数字的字号

    # 1. H2S出口浓度跟踪对比 (位置: 0, 0)
    ax1 = axes[0, 0]
    ax1.plot(results['fuzzy_pid']['time_steps'], results['fuzzy_pid']['predicted_outputs'],
             'b-', linewidth=line_width, label='Fuzzy Supervised PID')
    ax1.plot(results['traditional_pid']['time_steps'], results['traditional_pid']['predicted_outputs'],
             'g-', linewidth=line_width, label='Traditional PID')
    ax1.axhline(y=results['setpoint'], color='r', linestyle='--',
                linewidth=line_width, label='Target')
    ax1.set_xlabel('Time Step', fontsize=22)
    ax1.set_ylabel(r'$C_{\mathrm{out}} \, (\mathrm{mg/m}^3)$', fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=tick_label_size) # **修改点：加大刻度字号**
    ax1.legend(fontsize=18) # **保留此图的图例**

    # 2. 转速控制对比 (位置: 0, 1)
    ax2 = axes[0, 1]
    ax2.plot(results['fuzzy_pid']['time_steps'], results['fuzzy_pid']['rotation_speeds']*5,
             'b-', linewidth=line_width)
    ax2.plot(results['traditional_pid']['time_steps'], results['traditional_pid']['rotation_speeds']*5,
             'g-', linewidth=line_width)
    ax2.set_xlabel('Time Step', fontsize=22)
    ax2.set_ylabel('Rotation Speed (rpm)', fontsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=tick_label_size) # **修改点：加大刻度字号**
    # **修改点：删除此图的图例**

    # 3. 煤气进口流量控制对比 (位置: 1, 0)
    ax3 = axes[1, 0]
    ax3.plot(results['fuzzy_pid']['time_steps'], results['fuzzy_pid']['gas_flows'],
             'b-', linewidth=line_width)
    ax3.plot(results['traditional_pid']['time_steps'], results['traditional_pid']['gas_flows'],
             'g-', linewidth=line_width)
    ax3.set_xlabel('Time Step', fontsize=22)
    ax3.set_ylabel(r'$Flow_{\mathrm{gas}}\,(\mathrm{m^3/h})$', fontsize=22)
    ax3.tick_params(axis='both', which='major', labelsize=tick_label_size) # **修改点：加大刻度字号**
    # **修改点：删除此图的图例**

    # 4. 脱硫液流量控制对比 (位置: 1, 1)
    ax4 = axes[1, 1]
    ax4.plot(results['fuzzy_pid']['time_steps'], results['fuzzy_pid']['liquid_flows'],
             'b-', linewidth=line_width)
    ax4.plot(results['traditional_pid']['time_steps'], results['traditional_pid']['liquid_flows'],
             'g-', linewidth=line_width)
    ax4.set_xlabel('Time Step', fontsize=22)
    # **修改点：修正Y轴标签为液体流量**
    ax4.set_ylabel(r'$Flow_{\mathrm{liquid}}\,(\mathrm{m^3/h})$', fontsize=22)
    ax4.tick_params(axis='both', which='major', labelsize=tick_label_size) # **修改点：加大刻度字号**
    # **修改点：删除此图的图例**

    # 自动调整子图间距，确保布局紧凑美观
    plt.tight_layout(pad=3.0)

    # 保存为新文件
    plt.savefig('./pic/pid_control_variables_comparison.png',
                format='png', bbox_inches='tight', dpi=300)
    plt.savefig("./pic/pid_control_variables_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    print("PID关键变量对比结果图已保存到 ./pic/pid_control_variables_comparison.png 和 .pdf")



# ===================== MODIFICATION END =====================


def main():
    """主程序 - 支持加载模型和PID对比"""
    print(f"{'=' * 80}")
    print(f"S-ANFIS模糊监督PID控制系统")
    print(f"{'=' * 80}")

    # 检查是否有已训练的模型
    model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')] if os.path.exists(save_dir) else []

    if model_files:
        print(f"\n找到已训练的模型文件:")
        for i, file in enumerate(model_files):
            print(f"{i + 1}. {file}")

        choice = input("\n是否使用已有模型？(y/n): ").strip().lower()

        if choice == 'y':
            file_idx = int(input("请选择模型文件序号: ")) - 1
            model_path = os.path.join(save_dir, model_files[file_idx])
            trained_model, checkpoint = load_sanfis_model(model_path)
            # 从checkpoint中获取scaler
            scaler_X = checkpoint['scaler_X']
            scaler_y = checkpoint['scaler_y']
        else:
            # 训练新模型
            trained_model, model_mse, model_r2, membfuncs_config = train_sanfis_model()
            model_save_path = save_sanfis_model(trained_model, membfuncs_config, model_mse, model_r2)
    else:
        # 训练新模型
        trained_model, model_mse, model_r2, membfuncs_config = train_sanfis_model()
        model_save_path = save_sanfis_model(trained_model, membfuncs_config, model_mse, model_r2)

    # 创建控制器
    print(f"\n{'=' * 50}")
    print("创建控制器")
    print(f"{'=' * 50}")

    fuzzy_controller = FuzzySupervisedPIDController(
        trained_model, scaler_X, scaler_y, input_features
    )

    traditional_controller = TraditionalPIDController(
        kp=0.5, ki=0.02, kd=0.01
    )

    # 设置仿真参数
    setpoint = 1.0
    initial_conditions = {
        'gas_flow': 700.0,
        'liquid_flow': 18.0,
        'rotation_speed': 55.0
    }

    # 创建更平缓的进口H2S浓度变化曲线
    simulation_steps = 50
    base_inlet_h2s = 1000.0
    inlet_h2s_profile = base_inlet_h2s + 50 * np.sin(np.linspace(0, 2 * np.pi, simulation_steps)) + \
                        20 * np.random.normal(0, 0.1, simulation_steps)
    inlet_h2s_profile = np.clip(inlet_h2s_profile, 800, 1200)

    # 执行对比仿真
    comparison_results = simulate_control_comparison(
        fuzzy_controller, traditional_controller, trained_model,
        scaler_X, scaler_y, setpoint, initial_conditions,
        inlet_h2s_profile, simulation_steps
    )

    # 绘制对比结果
    plot_control_comparison(comparison_results)

    # 性能分析
    print(f"\n{'=' * 50}")
    print("控制性能对比分析")
    print(f"{'=' * 50}")

    # 模糊PID性能
    fuzzy_mae = np.mean(np.abs(comparison_results['fuzzy_pid']['errors']))
    fuzzy_max_error = np.max(np.abs(comparison_results['fuzzy_pid']['errors']))
    fuzzy_std = np.std(comparison_results['fuzzy_pid']['predicted_outputs'])

    # 传统PID性能
    traditional_mae = np.mean(np.abs(comparison_results['traditional_pid']['errors']))
    traditional_max_error = np.max(np.abs(comparison_results['traditional_pid']['errors']))
    traditional_std = np.std(comparison_results['traditional_pid']['predicted_outputs'])

    print(f"模糊监督PID:")
    print(f"  平均绝对误差: {fuzzy_mae:.3f} mg/m³")
    print(f"  最大绝对误差: {fuzzy_max_error:.3f} mg/m³")
    print(f"  输出标准差: {fuzzy_std:.3f} mg/m³")

    print(f"\n传统PID:")
    print(f"  平均绝对误差: {traditional_mae:.3f} mg/m³")
    print(f"  最大绝对误差: {traditional_max_error:.3f} mg/m³")
    print(f"  输出标准差: {traditional_std:.3f} mg/m³")

    print(f"\n性能改善:")
    print(f"  误差减少: {((traditional_mae - fuzzy_mae) / traditional_mae * 100):.1f}%")
    print(f"  稳定性改善: {((traditional_std - fuzzy_std) / traditional_std * 100):.1f}%")

    print(f"\n{'=' * 80}")
    print(f"程序执行完成！")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
