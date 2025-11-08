import os
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning


# -------------------- 模糊控制器类 --------------------
class FuzzyController:
    def __init__(self, setpoint):
        self.setpoint = setpoint

        # 定义模糊变量和论域
        self.error = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'error')
        self.delta_error = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), 'delta_error')
        self.output_adj = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'output_adj')

        # 自动生成隶属度函数
        self.error.automf(5, names=['NB', 'NS', 'ZO', 'PS', 'PB'])
        self.delta_error.automf(5, names=['NB', 'NS', 'ZO', 'PS', 'PB'])
        self.output_adj.automf(5, names=['NB', 'NS', 'ZO', 'PS', 'PB'])

        # 模糊规则库
        self.rules = [
            # 误差负大时的规则
            ctrl.Rule(self.error['NB'] & self.delta_error['NB'], self.output_adj['PB']),
            ctrl.Rule(self.error['NB'] & self.delta_error['NS'], self.output_adj['PB']),
            ctrl.Rule(self.error['NB'] & self.delta_error['ZO'], self.output_adj['PS']),

            # 误差负小时的规则
            ctrl.Rule(self.error['NS'] & self.delta_error['NB'], self.output_adj['PS']),
            ctrl.Rule(self.error['NS'] & self.delta_error['NS'], self.output_adj['PS']),

            # 误差接近零时的规则
            ctrl.Rule(self.error['ZO'] & self.delta_error['ZO'], self.output_adj['ZO']),

            # 误差正小时的规则
            ctrl.Rule(self.error['PS'] & self.delta_error['PS'], self.output_adj['NS']),
            ctrl.Rule(self.error['PS'] & self.delta_error['PB'], self.output_adj['NB']),

            # 误差正大时的规则
            ctrl.Rule(self.error['PB'] & self.delta_error['ZO'], self.output_adj['NB']),
            ctrl.Rule(self.error['PB'] & self.delta_error['PS'], self.output_adj['NB']),
        ]

        # 创建控制系统
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)
        self._last_error = 0  # 初始化上一次误差

    def update(self, current_value, dt):
        """更新模糊控制器并返回调整量"""
        error = self.setpoint - current_value
        delta_error = (error - self._last_error) / dt if dt != 0 else 0

        # 设置输入值
        self.simulator.input['error'] = error
        self.simulator.input['delta_error'] = delta_error

        try:
            self.simulator.compute()
            output = self.simulator.output['output_adj']
        except:
            output = 0  # 避免因未覆盖的输入范围导致报错

        self._last_error = error
        return output


# -------------------- 数据预处理部分 --------------------
def load_and_preprocess_data(data_file, sheet_name, x_columns, y_column, random_state=42):
    print(">> 读取并预处理数据...")
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    df.fillna(df.mean(), inplace=True)

    X = df[x_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return df, X_train, X_test, y_train, y_test, scaler


# -------------------- 模型训练部分 --------------------
def train_or_load_model(model_file, X_train_scaled, y_train):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if os.path.exists(model_file):
        print(f">> 加载现有模型: {model_file}")
        return joblib.load(model_file)
    else:
        print(">> 训练新模型...")
        mlp = MLPRegressor(
            hidden_layer_sizes=(50, 30),
            solver='adam',
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train)
        joblib.dump(mlp, model_file)
        return mlp


# -------------------- 模型评估部分 --------------------
def evaluate_model(mlp, X_test_scaled, y_test):
    y_pred = mlp.predict(X_test_scaled)
    print("=== 模型评估 ===")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²  : {r2_score(y_test, y_pred):.4f}")
    return y_pred


# -------------------- 模糊控制执行部分 --------------------
def fuzzy_control(df, mlp, scaler, x_columns, target, out_file, steps=500):
    print("\n>> 初始化模糊控制系统...")
    controller = FuzzyController(setpoint=target)

    # 初始控制变量
    control_vars = {
        'gas_flow': 996,
        'liquid_flow': 22,
        'frequency': 40
    }

    # 记录数据
    history = {
        'time': [],
        'h2s': [],
        'gas_flow': [],
        'liquid_flow': [],
        'frequency': []
    }

    # 变量约束范围
    var_limits = {
        'gas_flow': (df['gas_flow'].min(), df['gas_flow'].max()),
        'liquid_flow': (df['liquid_flow'].min(), df['liquid_flow'].max()),
        'frequency': (df['frequency'].min(), df['frequency'].max())
    }

    # 控制循环
    for step in range(steps):
        # 构造输入数据
        input_data = pd.DataFrame([[
            control_vars['gas_flow'],
            control_vars['liquid_flow'],
            control_vars['frequency']
        ]], columns=x_columns)

        # 预测当前状态
        current_h2s = mlp.predict(scaler.transform(input_data))[0]

        # 记录状态
        history['time'].append(step)
        history['h2s'].append(current_h2s)
        history['gas_flow'].append(control_vars['gas_flow'])
        history['liquid_flow'].append(control_vars['liquid_flow'])
        history['frequency'].append(control_vars['frequency'])

        # 获取控制量
        adjustment = controller.update(current_h2s, dt=1)

        # 调整控制变量（采用差异化的调整系数）
        control_vars['gas_flow'] += adjustment * 0.8
        control_vars['liquid_flow'] += adjustment * 1.2
        control_vars['frequency'] += adjustment * 0.5

        # 应用约束
        for var in control_vars:
            control_vars[var] = np.clip(
                control_vars[var],
                var_limits[var][0],
                var_limits[var][1]
            )

        # 显示状态
        if step % 50 == 0:
            print(f"Step {step}: H2S = {current_h2s:.2f} | "
                  f"Gas: {control_vars['gas_flow']:.1f}, "
                  f"Liquid: {control_vars['liquid_flow']:.1f}, "
                  f"Freq: {control_vars['frequency']:.1f}")

    # 保存结果
    result_df = pd.DataFrame(history)
    result_df.to_excel(out_file, index=False)
    print(f"\n>> 控制过程已保存到 {out_file}")

    # 可视化结果
    visualize_control(result_df, target)

    return result_df


# -------------------- 可视化函数 --------------------
def visualize_control(df, target):
    plt.figure(figsize=(14, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # H2S浓度变化
    plt.subplot(4, 1, 1)
    plt.plot(df['time'], df['h2s'], label='当前浓度', color='blue')
    plt.axhline(y=target, color='red', linestyle='--', label='目标浓度')
    plt.ylabel('H2S浓度 (mg/m³)')
    plt.title('模糊控制过程可视化')
    plt.legend()

    # 煤气流量变化
    plt.subplot(4, 1, 2)
    plt.plot(df['time'], df['gas_flow'], color='green')
    plt.ylabel('煤气流量')

    # 液体流量变化
    plt.subplot(4, 1, 3)
    plt.plot(df['time'], df['liquid_flow'], color='purple')
    plt.ylabel('液体流量')

    # 频率变化
    plt.subplot(4, 1, 4)
    plt.plot(df['time'], df['frequency'], color='orange')
    plt.ylabel('频率')
    plt.xlabel('时间步')

    plt.tight_layout()
    plt.show()


# -------------------- 主函数 --------------------
def main():
    # 配置参数
    config = {
        'data_file': './data/dataset.xlsx',
        'model_file': './model/h2s_model.pkl',
        'sheet_name': 'Sheet2',
        'x_columns': ['gas_flow', 'liquid_flow', 'frequency'],
        'y_column': 'H2S_concentration',
        'control_target': 20.0,
        'output_file': './results/fuzzy_control.xlsx'
    }

    # 数据预处理
    df, X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        config['data_file'],
        config['sheet_name'],
        config['x_columns'],
        config['y_column']
    )

    # 模型训练/加载
    model = train_or_load_model(config['model_file'], scaler.transform(X_train), y_train)

    # 模型评估
    evaluate_model(model, scaler.transform(X_test), y_test)

    # 执行模糊控制
    fuzzy_control(
        df=df,
        mlp=model,
        scaler=scaler,
        x_columns=config['x_columns'],
        target=config['control_target'],
        out_file=config['output_file'],
        steps=500
    )


if __name__ == "__main__":
    main()