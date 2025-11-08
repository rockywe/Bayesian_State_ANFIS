import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning


# 定义PID控制器类
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(None, None)):
        """
        初始化PID控制器
        :param Kp: 比例系数
        :param Ki: 积分系数
        :param Kd: 微分系数
        :param setpoint: 目标设定值
        :param output_limits: 输出限制 (最小值, 最大值)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._last_error = 0
        self._integral = 0

    def update(self, current_value, dt):
        """
        更新PID控制器并计算控制输出
        :param current_value: 当前测量值
        :param dt: 时间步长
        :return: 控制输出
        """
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._last_error) / dt if dt > 0 else 0

        # PID输出
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        # 应用输出限制
        lower, upper = self.output_limits
        if lower is not None:
            output = max(lower, output)
        if upper is not None:
            output = min(upper, output)

        # 保存用于下一次迭代
        self._last_error = error

        return output


# 加载和预处理数据
def load_and_preprocess_data(data_file, sheet_name, x_columns, y_column, random_state=42):
    # 读取Excel数据
    print(">> 读取Excel数据...")
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    # 简单缺失值处理
    df.fillna(df.mean(), inplace=True)

    # 特征和目标
    X = df[x_columns]
    y = df[y_column]

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return df, X_train, X_test, y_train, y_test, scaler


# 训练或加载模型
def train_or_load_model(model_file, X_train_scaled, y_train):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if os.path.exists(model_file):
        print(f">> 检测到已有模型: {model_file}, 正在加载...")
        mlp = joblib.load(model_file)
    else:
        print(">> 未检测到模型文件，开始训练 MLPRegressor...")
        mlp = MLPRegressor(
            hidden_layer_sizes=(50, 30),
            solver='adam',
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train)

        print(">> 训练完成，保存模型到文件...")
        joblib.dump(mlp, model_file)

    return mlp


# 评估模型
def evaluate_model(mlp, X_test_scaled, y_test):
    y_pred = mlp.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("=== 测试集评估 ===")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    return y_pred


# 预测并保存结果
def predict_and_save(df, mlp, scaler, x_columns, out_excel):
    print(">> 在整份数据上进行预测...")
    X_all = df[x_columns]
    X_all_scaled = scaler.transform(X_all)
    y_all_pred = mlp.predict(X_all_scaled)

    # 将预测结果存入 DataFrame
    df['Predicted_H2S_concentration'] = y_all_pred

    # 将新列保存到新的 Excel 文件
    df.to_excel(out_excel, sheet_name='Sheet1', index=False)
    print(f">> 已将预测结果写入 {out_excel}，列名为 'Predicted_H2S_concentration'")

    return df


# 绘制模型预测结果
def plot_results(df, y_column, pred_column):
    # 设置中文字体以避免字体警告
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或其他支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False

    # 真实值 vs. 预测值
    print(">> 绘图: 真实值 vs. 预测值 (整体数据) ...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[y_column], y=df[pred_column], color='blue', edgecolor='w', s=70)

    min_val = min(df[y_column].min(), df[pred_column].min())
    max_val = max(df[y_column].max(), df[pred_column].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x参考线')

    plt.xlabel('真实值', fontsize=14)
    plt.ylabel('预测值', fontsize=14)
    plt.title('真实值 vs. 预测值(整体数据)', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 残差分布
    print(">> 绘图: 残差分布 ...")
    residuals = df[y_column] - df[pred_column]

    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True, color='green')
    plt.xlabel('残差 (真实值 - 预测值)', fontsize=14)
    plt.ylabel('频数', fontsize=14)
    plt.title('残差分布(整体数据)', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# PID控制循环
def pid_control(df, mlp, scaler, x_columns, y_column, target, pid_params, out_excel_control, steps=100):
    # pid_params 应为字典，包含 'Kp', 'Ki', 'Kd'
    Kp = pid_params['Kp']
    Ki = pid_params['Ki']
    Kd = pid_params['Kd']

    # 初始化PID控制器
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=target, output_limits=(-10, 10))  # 输出限制可根据需要调整

    # 初始化控制变量，确保在训练数据范围内
    gas_flow_mean = df['gas_flow'].mean()
    liquid_flow_mean = df['liquid_flow'].mean()
    frequency_mean = df['frequency'].mean()

    # current_gas_flow = gas_flow_mean
    # current_liquid_flow = liquid_flow_mean
    # current_frequency = frequency_mean

    current_gas_flow = 996
    current_liquid_flow = 22
    current_frequency = 40

    # 获取训练数据的范围，用于限制控制变量
    gas_flow_min, gas_flow_max = df['gas_flow'].min(), df['gas_flow'].max()
    liquid_flow_min, liquid_flow_max = df['liquid_flow'].min(), df['liquid_flow'].max()
    frequency_min, frequency_max = df['frequency'].min(), df['frequency'].max()

    # 初始化列表以记录模拟数据
    y_simulation = []
    gas_flow_simulation = []
    liquid_flow_simulation = []
    frequency_simulation = []
    time_simulation = []

    print(">> 开始PID控制循环...")

    for step in range(steps):
        current_time = step * 1  # 假设每步时间间隔为1单位时间

        # 准备当前输入特征，确保使用 DataFrame 并保持特征名称
        current_input = pd.DataFrame({
            'gas_flow': [current_gas_flow],
            'liquid_flow': [current_liquid_flow],
            'frequency': [current_frequency]
        })
        current_input_scaled = scaler.transform(current_input)

        # 使用神经网络预测当前H2S浓度
        y_current = mlp.predict(current_input_scaled)[0]

        # 记录模拟数据
        y_simulation.append(y_current)
        gas_flow_simulation.append(current_gas_flow)
        liquid_flow_simulation.append(current_liquid_flow)
        frequency_simulation.append(current_frequency)
        time_simulation.append(step * 1)

        # 计算误差并获取控制动作
        control_action = pid.update(y_current, dt=1)  # dt=1单位时间

        # 调整控制变量基于控制动作
        # 这里采用将控制动作均匀分配给三个变量，比例可根据实际情况调整
        adjustment_factor = 0.1  # 调整因子，避免一次性调整过大

        current_gas_flow += control_action * adjustment_factor
        current_liquid_flow += control_action * adjustment_factor
        current_frequency += control_action * adjustment_factor

        # 强制限制控制变量在训练数据范围内
        current_gas_flow = max(gas_flow_min, min(current_gas_flow, gas_flow_max))
        current_liquid_flow = max(liquid_flow_min, min(current_liquid_flow, liquid_flow_max))
        current_frequency = max(frequency_min, min(current_frequency, frequency_max))

        print(
            f"Step: {step}, H2S: {y_current:.2f} mg/m³, Gas Flow: {current_gas_flow:.2f}, Liquid Flow: {current_liquid_flow:.2f}, Frequency: {current_frequency:.2f}")

        # 可选：停止条件
        if abs(target - y_current) < 0.1:  # 根据需要调整阈值
            print(">> 目标浓度已达到，停止控制。")
            break

    # 将控制过程数据存入DataFrame
    control_df = pd.DataFrame({
        'Time': time_simulation,
        'H2S_concentration': y_simulation,
        'Gas_flow': gas_flow_simulation,
        'Liquid_flow': liquid_flow_simulation,
        'Frequency': frequency_simulation
    })

    # 保存控制过程数据到Excel
    control_df.to_excel(out_excel_control, sheet_name='PID_Control', index=False)
    print(f">> 已将PID控制过程数据保存到 {out_excel_control}")

    # 绘制控制效果图
    plot_pid_control(control_df, target)


def plot_pid_control(control_df, target):
    # 设置中文字体以避免字体警告
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或其他支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(14, 12))

    # H2S浓度
    plt.subplot(4, 1, 1)
    plt.plot(control_df['Time'], control_df['H2S_concentration'], label='预测 H2S 浓度', color='blue')
    plt.axhline(y=target, color='r', linestyle='--', label='目标浓度')
    plt.xlabel('时间步')
    plt.ylabel('H2S浓度 (mg/m³)')
    plt.title('PID控制下的H2S浓度变化')
    plt.legend()
    plt.grid(True)

    # 进口煤气流量
    plt.subplot(4, 1, 2)
    plt.plot(control_df['Time'], control_df['Gas_flow'], label='进口煤气流量', color='green')
    plt.xlabel('时间步')
    plt.ylabel('进口煤气流量')
    plt.title('进口煤气流量变化')
    plt.legend()
    plt.grid(True)

    # 进口液流量
    plt.subplot(4, 1, 3)
    plt.plot(control_df['Time'], control_df['Liquid_flow'], label='进口液流量', color='purple')
    plt.xlabel('时间步')
    plt.ylabel('进口液流量')
    plt.title('进口液流量变化')
    plt.legend()
    plt.grid(True)

    # 频率
    plt.subplot(4, 1, 4)
    plt.plot(control_df['Time'], control_df['Frequency'], label='频率', color='orange')
    plt.xlabel('时间步')
    plt.ylabel('频率')
    plt.title('频率变化')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # 基础配置
    data_file = './data/dataset.xlsx'  # 您的数据文件路径
    model_file = './model/my_trained_mlp.pkl'  # 模型权重保存路径
    sheet_name = 'Sheet2'  # Excel中的工作表名称
    out_excel = './data/dataset_with_pred.xlsx'  # 预测结果保存路径
    out_excel_control = './data/pid_control_results.xlsx'  # PID控制过程数据保存路径

    # 特征及目标列名，根据实际Excel表格修改
    x_columns = ['gas_flow', 'liquid_flow', 'frequency']  # 举例：三维特征
    y_column = 'H2S_concentration'  # 举例：目标列

    # 加载和预处理数据
    df, X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        data_file, sheet_name, x_columns, y_column, random_state=42
    )

    # 训练或加载模型
    mlp = train_or_load_model(model_file, scaler.transform(X_train), y_train)

    # 评估模型
    y_pred = evaluate_model(mlp, scaler.transform(X_test), y_test)

    # 预测并保存结果
    df = predict_and_save(df, mlp, scaler, x_columns, out_excel)

    # 绘制模型预测结果
    plot_results(df, y_column, 'Predicted_H2S_concentration')

    # 定义PID控制器参数
    target = 20.0  # 目标H2S浓度 (mg/m³)，根据您的数据调整
    pid_params = {
        'Kp': 1.0,
        'Ki': 0.05,
        'Kd': 0.01
    }

    # 进行PID控制
    pid_control(
        df=df,
        mlp=mlp,
        scaler=scaler,
        x_columns=x_columns,
        y_column=y_column,
        target=target,
        pid_params=pid_params,
        out_excel_control=out_excel_control,
        steps=500
    )


if __name__ == "__main__":
    main()
