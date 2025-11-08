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


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._last_error = 0
        self._integral = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._last_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        lower, upper = self.output_limits
        if lower is not None:
            output = max(lower, output)
        if upper is not None:
            output = min(upper, output)

        self._last_error = error
        return output


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

    print("训练数据维度:", X_train_scaled.shape)  # 应为 (n, 8)

    return df, X_train, X_test, y_train, y_test, scaler


def train_or_load_model(model_file, X_train_scaled, y_train):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if os.path.exists(model_file):
        print(f">> 加载已有模型: {model_file}")
        return joblib.load(model_file)

    print(">> 训练新模型...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50, 30),  # 3层网络
        activation='swish',  # 使用Swish激活函数
        solver='nadam',  # 改进的优化器
        alpha=0.0001,  # L2正则化强度
        learning_rate='adaptive',  # 自适应学习率
        early_stopping=True,  # 早停机制
        validation_fraction=0.1,  # 验证集比例
        max_iter=5000,  # 增加最大迭代
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    joblib.dump(mlp, model_file)
    return mlp


def evaluate_model(mlp, X_test_scaled, y_test):
    y_pred = mlp.predict(X_test_scaled)
    print("\n=== 模型评估 ===")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²  : {r2_score(y_test, y_pred):.4f}")
    return y_pred


def predict_and_save(df, mlp, scaler, x_columns, out_excel):
    print("\n>> 全量数据预测...")
    X_all_scaled = scaler.transform(df[x_columns])
    df['Predicted_Outlet_H2S'] = mlp.predict(X_all_scaled)
    df.to_excel(out_excel, index=False)
    print(f"预测结果已保存至: {out_excel}")
    return df


def plot_results(df, y_true, y_pred):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df[y_true], y=df[y_pred], alpha=0.6)
    plt.plot([df[y_true].min(), df[y_true].max()],
             [df[y_true].min(), df[y_true].max()], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('真实值 vs 预测值')

    plt.subplot(1, 2, 2)
    residuals = df[y_true] - df[y_pred]
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('残差')
    plt.title('残差分布')

    plt.tight_layout()
    plt.show()


def pid_control(df, mlp, scaler, x_columns, target, pid_params, out_excel, steps=200):
    # 可调节参数配置（控制变量）
    control_vars = {
        '煤气进口流量': {
            'value': df['煤气进口流量'].mean(),
            'range': (df['煤气进口流量'].min(), df['煤气进口流量'].max())
        },
        '脱硫液流量': {
            'value': df['脱硫液流量'].mean(),
            'range': (df['脱硫液流量'].min(), df['脱硫液流量'].max())
        },
        '转速': {
            'value': df['转速'].mean(),
            'range': (df['转速'].min(), df['转速'].max())
        }
    }

    # 固定参数配置（取数据平均值）
    fixed_params = {
        '进口煤气温度': df['进口煤气温度'].mean(),
        '进口煤气压力': df['进口煤气压力'].mean(),
        '脱硫液温度': df['脱硫液温度'].mean(),
        '脱硫液压力': df['脱硫液压力'].mean(),
        '进口H2S浓度': df['进口H2S浓度'].mean()
    }

    # 初始化PID控制器
    pids = {
        var: PIDController(
            Kp=pid_params['Kp'],
            Ki=pid_params['Ki'],
            Kd=pid_params['Kd'],
            setpoint=target,
            output_limits=(-5, 5)
        ) for var in control_vars
    }

    records = []
    print("\n>> PID控制启动，可调节参数：煤气进口流量、脱硫液流量、转速")

    for step in range(steps):
        # 构造输入特征
        input_data = { ** {var: control_vars[var]['value'] for var in control_vars}, ** fixed_params}
        input_df = pd.DataFrame([input_data]).reindex(columns=x_columns)

        # 预测当前状态
        X_scaled = scaler.transform(input_df)
        current_h2s = mlp.predict(X_scaled)[0]

        # 记录数据
        records.append({
            'step': step,
            'outlet_H2S': current_h2s,
            ** control_vars,
            **fixed_params
        })

        # 更新控制参数
        for var in control_vars:
            adjustment = pids[var].update(current_h2s, dt=1)
            new_value = control_vars[var]['value'] + adjustment * 0.1
            control_vars[var]['value'] = np.clip(new_value, *control_vars[var]['range'])

        # 显示进度
        if step % 50 == 0:
            print(f"Step {step}: 出口H2S = {current_h2s:.2f} mg/m³")

    # 保存控制过程
    control_df = pd.DataFrame(records)
    control_df.to_excel(out_excel, index=False)
    print(f"控制过程已保存至: {out_excel}")

    # 绘制控制效果
    plt.figure(figsize=(10, 6))
    plt.plot(control_df['step'], control_df['outlet_H2S'], label='出口H2S')
    plt.axhline(target, color='r', linestyle='--', label='目标值')
    plt.xlabel('控制步数')
    plt.ylabel('H2S浓度 (mg/m³)')
    plt.title('PID控制过程')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # 配置参数
    config = {
        'data_file': './data2/脱硫数据整理.xlsx',
        'model_file': './model/pid_nn2.pkl',
        'sheet_name': 'Sheet1',
        'output_pred': './data2/predictions.xlsx',
        'output_control': './data2/control_log.xlsx',
        'x_columns': [
            '煤气进口流量',  # 煤气进口流量
            '进口煤气温度',  # 进口煤气温度
            '进口煤气压力',  # 进口煤气压力
            '脱硫液流量',  # 脱硫液流量
            '脱硫液温度',  # 脱硫液温度
            '脱硫液压力',  # 脱硫液压力
            '转速',  # 转速
            '进口H2S浓度'  # 进口H2S浓度
        ],
        'y_column': '出口H2S浓度',  # 出口H2S浓度
        'pid_params': {
            'Kp': 1.2,
            'Ki': 0.05,
            'Kd': 0.1
        },
        'target_h2s': 20.0  # 目标出口H2S浓度
    }

    # 数据预处理
    df, X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        config['data_file'], config['sheet_name'],
        config['x_columns'], config['y_column']
    )

    # 模型训练/加载
    model = train_or_load_model(
        model_file=config['model_file'],  # 正确键名
        X_train_scaled=scaler.transform(X_train),  # 添加参数名明确对应
        y_train=y_train
    )

    # 模型评估
    evaluate_model(model, scaler.transform(X_test), y_test)

    # 全量预测
    df_pred = predict_and_save(df, model, scaler, config['x_columns'], config['output_pred'])

    # 结果可视化
    plot_results(df_pred, config['y_column'], 'Predicted_Outlet_H2S')

    # PID控制
    pid_control(
        df=df,
        mlp=model,
        scaler=scaler,
        x_columns=config['x_columns'],
        target=config['target_h2s'],
        pid_params=config['pid_params'],
        out_excel=config['output_control'],
        steps=300
    )
if __name__ == "__main__":
        main()