# =============================================================================
#  测试脚本 (test_hybrid.py) - 使用 HybridANFIS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from sklearn.metrics import mean_squared_error
import shap

from anfis_test.hybrid_anfis_model import HybridANFIS


def run_test():
    # ========== 1) 读取数据 ==========
    file_path = './data/脱硫数据整理.xlsx'
    try:
        data_df = pd.read_excel(file_path, sheet_name='Sheet1') # Or your sheet name
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    input_features = [
        '煤气进口流量', '进口煤气温度', '进口煤气压力',
        '脱硫液流量', '脱硫液温度', '脱硫液压力',
        '转速', '进口H2S浓度'
    ]
    output_feature = '出口H2S浓度'

    missing_cols = [col for col in input_features + [output_feature] if col not in data_df.columns]
    if missing_cols:
        print(f"Error: Missing columns in data: {', '.join(missing_cols)}")
        return

    X = data_df[input_features].values
    y = data_df[output_feature].values.reshape(-1, 1)
    print(f"Original data shape: X={X.shape}, y={y.shape}")

    # ========== 2) 处理缺失值 ==========
    if np.isnan(X).any() or np.isnan(y).any():
        print("Handling missing values using column means...")
        col_means_X = np.nanmean(X, axis=0)
        col_mean_y = np.nanmean(y)
        inds_X = np.where(np.isnan(X))
        inds_y = np.where(np.isnan(y))
        X[inds_X] = np.take(col_means_X, inds_X[1])
        y[inds_y] = col_mean_y
        print("Missing values handled.")
    else:
        print("No missing values detected.")

    # ========== 3) 划分数据集 ==========
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    print(f"Data split: Train={X_train.shape}, Validation={X_val.shape}, Test={X_test.shape}")

    # ========== 4) 标准化特征 X ==========
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    print("Features X standardized.")
    # Note: y scaling is now handled *inside* the HybridANFIS class

    # ========== 5) 构建 HybridANFIS 模型 ==========
    n_input = X_train_scaled.shape[1]
    n_mfs = 2
    # NEW: Set a value for LSE regularization lambda
    lse_regularization_strength = 1e-3 # Needs tuning
    anfis_model = HybridANFIS(n_input=n_input, n_mfs=n_mfs, lse_lambda=lse_regularization_strength)

    # ========== 6) 训练模型 ==========
    epochs = 20
    # NEW: Try a smaller learning rate for premise parameters first
    learning_rate_premise = 0.01 # Needs tuning
    early_stopping_patience = 30

    history = anfis_model.fit(
        X_train_scaled, y_train, # Pass original y, scaling happens inside fit
        epochs=epochs,
        lr_premise=learning_rate_premise,
        validation_data=(X_val_scaled, y_val), # Pass original y_val
        early_stopping_patience=early_stopping_patience
    )

    # ========== 7) 绘制损失曲线 ==========
    plt.figure(figsize=(10, 5))
    # Plot MSE based on scaled values as reported during training
    plt.plot(history['train_loss'], label='Training MSE (scaled)')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation MSE (scaled)')
    plt.title('Hybrid ANFIS Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE) - Scaled')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Keep log scale
    plt.show()

    # ========== 8) 预测与评估 ==========
    print("\n--- Evaluating on Test Set ---")
    # NEW: Get predictions in original scale by default
    y_pred = anfis_model.predict(X_test_scaled)
    y_test_orig = y_test.flatten() # Original test targets

    # plot_shap(anfis_model, X_train_scaled, X_test_scaled, input_features)
    plot_error_distribution(y_test_orig, y_pred)
    plot_training_loss(history)
    plot_noise_impact(anfis_model, X_test_scaled, y_test_orig)

    # Calculate metrics on original scale
    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred)

    print("Metrics calculated on original scale:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    # ========== 9) 绘制实际 vs 预测 (原始尺度) ==========
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_orig, y=y_pred, color='dodgerblue', edgecolor='k', alpha=0.7, s=60, label='Predictions')
    min_val = min(y_test_orig.min(), y_pred.min()) * 0.9 # Adjust plot limits slightly
    max_val = max(y_test_orig.max(), y_pred.max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
    plt.xlabel(f'Actual {output_feature}', fontsize=14)
    plt.ylabel(f'Predicted {output_feature}', fontsize=14)
    plt.title('Hybrid ANFIS - Actual vs. Predicted (Test Set - Original Scale)', fontsize=16)
    plt.xlim(min_val, max_val) # Set adjusted limits
    plt.ylim(min_val, max_val)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # ========== 10) 残差分析 (原始尺度) ==========
    residuals = y_test_orig - y_pred
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_pred, y=residuals, color='mediumseagreen', edgecolor='k', alpha=0.7, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Value (Original Scale)', fontsize=12)
    plt.ylabel('Residuals (Original Scale)', fontsize=12)
    plt.title('Residuals vs. Predicted', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, bins=30, kde=True, color='salmon')
    plt.xlabel('Residuals (Original Scale)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residual Distribution', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Optional steps 11 & 12 (Rules, MFs) remain the same conceptually

def plot_shap(anfis_model, X_train_scaled, X_test_scaled, input_features):
    """生成 SHAP 图以解释特征重要性"""
    explainer = shap.KernelExplainer(anfis_model.predict, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=input_features, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig('shap_summary.png')
    plt.close()

def plot_error_distribution(y_test_orig, y_pred):
    """绘制误差分布图以展示模型性能"""
    errors = y_test_orig - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(errors, bins=30, kde=True, color='blue')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig('error_distribution.png')
    plt.close()

def plot_training_loss(history):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

def plot_noise_impact(anfis_model, X_test_scaled, y_test_orig):
    """绘制模型在不同噪声水平下的性能"""
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    mse_with_noise = []
    for noise in noise_levels:
        X_test_noisy = X_test_scaled + np.random.normal(0, noise, X_test_scaled.shape)
        y_pred_noisy = anfis_model.predict(X_test_noisy)
        mse = mean_squared_error(y_test_orig, y_pred_noisy)
        mse_with_noise.append(mse)
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, mse_with_noise, marker='o')
    plt.title('Model Performance with Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig('noise_impact.png')
    plt.close()


if __name__ == "__main__":
    run_test()
