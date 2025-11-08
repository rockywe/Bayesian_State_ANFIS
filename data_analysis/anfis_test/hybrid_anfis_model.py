# hybrid_anfis_model.py
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import time
# NEW: Import StandardScaler for y scaling
from sklearn.preprocessing import StandardScaler

# --- 隶属函数 ---
def gaussian_mf(x, c, s):
    """
    高斯隶属函数
    x: 输入数组 (N,) 或单个值
    c: 高斯分布中心
    s: 标准差 (越大越平缓) - 必须为正
    返回: x 在该高斯模糊集中对应的隶属度
    """
    # 确保 s 是正数且不为零
    s = np.maximum(s, 1e-8)
    return np.exp(-0.5 * ((x - c) / s)**2)

# --- ANFIS 类 (混合学习 - 改进稳定性) ---
class HybridANFIS:
    # NEW: Added lse_lambda parameter for regularization
    def __init__(self, n_input, n_mfs, mf_func=gaussian_mf, lse_lambda=1e-4):
        """
        初始化混合学习 ANFIS 模型

        参数:
        - n_input: 输入维度 (特征数量)
        - n_mfs: 每个输入的模糊隶属函数数量
        - mf_func: 使用的隶属函数 (默认高斯)
        - lse_lambda: LSE 正则化强度 (Ridge Regression)
        """
        self.n_input = n_input
        self.n_mfs = n_mfs
        self.mf_func = mf_func
        self.lse_lambda = lse_lambda # Store regularization strength

        # 1) 前件参数 (Premise Parameters): centers, sigmas
        self.centers = np.random.uniform(-1.5, 1.5, size=(n_input, n_mfs))
        self.sigmas = np.random.uniform(0.5, 2.0, size=(n_input, n_mfs))

        # 2) 规则数量
        self.n_rules = (n_mfs ** n_input)
        print(f"Initializing HybridANFIS with {self.n_input} inputs, {self.n_mfs} MFs per input.")
        print(f"Total number of rules: {self.n_rules}")
        print(f"LSE Regularization Lambda: {self.lse_lambda}") # Print lambda
        if self.n_rules > 1000:
            print("Warning: High number of rules (>1000). Training might be slow and require significant memory.")

        # 3) 后件参数 (Consequent Parameters)
        self.consequents = np.random.randn(self.n_rules, self.n_input + 1) * 0.1

        # 4) 规则索引和隶属函数名称
        self._rule_indices = list(product(range(self.n_mfs), repeat=self.n_input))
        self.mf_names = self._generate_mf_names()

        if len(self._rule_indices) != self.n_rules:
             raise ValueError(f"Internal error: Mismatch in rule count.")

        # NEW: Scaler for the target variable y
        self.scaler_y = StandardScaler()
        self._y_scaled = False # Flag to track if y has been scaled during fit

    def _generate_mf_names(self):
        """生成隶属函数的名称 (例如 Low, High)"""
        mf_names = []
        for i in range(self.n_input):
            names = []
            for j in range(self.n_mfs):
                if self.n_mfs == 2:
                    mf_label = ["Low", "High"][j]
                elif self.n_mfs == 3:
                    mf_label = ["Small", "Medium", "Large"][j]
                else:
                    mf_label = f"MF{j}"
                names.append(mf_label)
            mf_names.append(names)
        return mf_names

    def _forward_pass_internal(self, X):
        """
        执行前向传播计算，返回中间结果用于训练。
        (与之前版本相同)
        """
        N = X.shape[0]
        mus = np.zeros((self.n_input, self.n_mfs, N))
        for i in range(self.n_input):
            for j in range(self.n_mfs):
                mus[i, j, :] = self.mf_func(X[:, i], self.centers[i, j], self.sigmas[i, j])

        rule_mus = np.ones((N, self.n_rules))
        for rule_idx, combo in enumerate(self._rule_indices):
            for input_idx, mf_idx in enumerate(combo):
                rule_mus[:, rule_idx] *= mus[input_idx, mf_idx, :]

        rule_sum = np.sum(rule_mus, axis=1, keepdims=True)
        rule_sum[rule_sum == 0] = 1e-10
        rule_mus_norm = rule_mus / rule_sum

        X_plus = np.hstack([X, np.ones((N, 1))])
        rule_outputs = X_plus @ self.consequents.T
        y_pred = np.sum(rule_mus_norm * rule_outputs, axis=1)
        return mus, rule_mus, rule_mus_norm, rule_outputs, y_pred

    # MODIFIED: Added Regularization (Ridge)
    def _update_consequents_lse_ridge(self, X, y_scaled, rule_mus_norm):
        """
        使用带 L2 正则化的最小二乘估计 (Ridge Regression) 更新后件参数。

        参数:
        - X: 输入数据, shape (N, n_input)
        - y_scaled: 标准化后的目标输出, shape (N, 1)
        - rule_mus_norm: 归一化规则激活度, shape (N, n_rules)
        """
        N = X.shape[0]
        X_plus = np.hstack([X, np.ones((N, 1))]) # shape (N, n_input + 1)
        num_params_per_rule = self.n_input + 1
        total_consequent_params = self.n_rules * num_params_per_rule

        # 构造 LSE 的系数矩阵 A (与之前相同)
        A = np.zeros((N, total_consequent_params))
        for k in range(self.n_rules):
            w_norm_k = rule_mus_norm[:, k:k+1]
            A_k = w_norm_k * X_plus
            start_col = k * num_params_per_rule
            end_col = start_col + num_params_per_rule
            A[:, start_col:end_col] = A_k

        # --- Ridge Regression ---
        # 求解 (A^T A + lambda * I) * p = A^T * y
        # 其中 I 是单位矩阵, lambda 是正则化强度 (self.lse_lambda)
        AtA = A.T @ A # Shape: (total_params, total_params)
        AtY = A.T @ y_scaled # Shape: (total_params, 1)

        # 添加正则化项 lambda * I
        identity_matrix = np.identity(total_consequent_params)
        regularized_AtA = AtA + self.lse_lambda * identity_matrix

        try:
            # 解线性方程组 (Regularized)
            p_vec = np.linalg.solve(regularized_AtA, AtY)
            # 更新后件参数
            self.consequents = p_vec.reshape(self.n_rules, num_params_per_rule)
        except np.linalg.LinAlgError as e:
            print(f"Warning: Ridge LSE failed ({e}). Trying pseudo-inverse.")
            # 如果 solve 失败 (例如矩阵仍然接近奇异)，尝试使用伪逆 (pinv)
            try:
                pseudo_inv = np.linalg.pinv(regularized_AtA)
                p_vec = pseudo_inv @ AtY
                self.consequents = p_vec.reshape(self.n_rules, num_params_per_rule)
            except np.linalg.LinAlgError as e2:
                 print(f"Warning: Pseudo-inverse also failed ({e2}). Skipping consequent update.")


    # MODIFIED: Uses y_scaled for error calculation
    def _backward_premise_gradient(self, X, y_pred_scaled, y_true_scaled, mus, rule_mus, rule_mus_norm, rule_outputs):
        """
        计算误差关于前件参数 (centers, sigmas) 的梯度。
        现在使用标准化后的 y 进行误差计算。

        参数:
        - X: 输入数据, shape (N, n_input)
        - y_pred_scaled: 标准化后的预测输出, shape (N,)
        - y_true_scaled: 标准化后的真实输出, shape (N,)
        - mus: 隶属度, shape (n_input, n_mfs, N)
        - rule_mus: 未归一化规则激活度, shape (N, n_rules)
        - rule_mus_norm: 归一化规则激活度, shape (N, n_rules)
        - rule_outputs: 规则后件输出 (基于标准化尺度), shape (N, n_rules)

        返回:
        - grad_centers: centers 的梯度, shape (n_input, n_mfs)
        - grad_sigmas: sigmas 的梯度, shape (n_input, n_mfs)
        """
        N = X.shape[0]
        # 使用标准化后的 y 计算误差
        error_scaled = y_pred_scaled.reshape(N, 1) - y_true_scaled.reshape(N, 1)
        grad_centers = np.zeros_like(self.centers)
        grad_sigmas = np.zeros_like(self.sigmas)

        # --- 准备链式法则的中间项 ---
        # dE/dy_pred_scaled = 2/N * error_scaled
        dE_dypred = (2.0 / N) * error_scaled # shape (N, 1)

        # (其余计算与之前版本类似，但基于 scaled error 和 scaled rule_outputs)
        sum_w = np.sum(rule_mus, axis=1, keepdims=True)
        sum_w_sq = sum_w**2
        sum_w_sq[sum_w_sq == 0] = 1e-10

        # --- 计算梯度 ---
        for i in range(self.n_input):
            for j in range(self.n_mfs):
                mu_ij = mus[i, j, :]
                c_ij = self.centers[i, j]
                s_ij = self.sigmas[i, j]
                s_ij = np.maximum(s_ij, 1e-8)

                dmu_dc = mu_ij * (X[:, i] - c_ij) / (s_ij**2)
                dmu_ds = mu_ij * ((X[:, i] - c_ij)**2) / (s_ij**3)

                sum_dE_dc = 0.0
                sum_dE_ds = 0.0
                relevant_rule_indices = [k for k, combo in enumerate(self._rule_indices) if combo[i] == j]

                for k in relevant_rule_indices:
                    w_k = rule_mus[:, k]
                    f_k = rule_outputs[:, k] # 注意：这里的 f_k 是基于 scaled y 的

                    mu_ij_safe = np.copy(mu_ij)
                    mu_ij_safe[mu_ij_safe == 0] = 1e-10
                    dwk_dmuij = w_k / mu_ij_safe

                    temp_sum = np.zeros(N)
                    for l in range(self.n_rules):
                        w_l = rule_mus[:, l]
                        f_l = rule_outputs[:, l] # 注意：这里的 f_l 是基于 scaled y 的
                        delta_lk = 1.0 if l == k else 0.0
                        dw_norml_dwk = (delta_lk * sum_w.flatten() - w_l) / sum_w_sq.flatten()
                        temp_sum += f_l * dw_norml_dwk

                    dypred_dmuij = temp_sum * dwk_dmuij
                    sum_dE_dc += np.sum(dE_dypred.flatten() * dypred_dmuij * dmu_dc)
                    sum_dE_ds += np.sum(dE_dypred.flatten() * dypred_dmuij * dmu_ds)

                grad_centers[i, j] = sum_dE_dc
                grad_sigmas[i, j] = sum_dE_ds

        return grad_centers, grad_sigmas

    # MODIFIED: Handles y scaling and uses Ridge LSE
    def fit(self, X, y, epochs=100, lr_premise=0.01, validation_data=None, early_stopping_patience=None):
        """
        使用混合学习算法训练 ANFIS 模型 (包含 y 标准化和 LSE 正则化)。
        """
        y = y.reshape(-1, 1)
        # NEW: Scale y using the fitted scaler
        y_scaled = self.scaler_y.fit_transform(y)
        self._y_scaled = True # Mark y as scaled

        # Prepare validation data if provided
        X_val_scaled, y_val_scaled = None, None
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val = y_val.reshape(-1, 1)
            # NEW: Scale validation y using the *same* scaler fitted on training y
            y_val_scaled = self.scaler_y.transform(y_val)

        best_val_loss = np.inf
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        print(f"\n--- Starting Hybrid Training (Improved Stability) ---")
        print(f"Epochs: {epochs}, Premise LR: {lr_premise}, LSE Lambda: {self.lse_lambda}")
        if validation_data:
            print(f"Using Validation Set. Early Stopping Patience: {early_stopping_patience}")

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # --- 前向传播 ---
            mus, rule_mus, rule_mus_norm, _, _ = self._forward_pass_internal(X)

            # --- 更新后件参数 (Ridge LSE) ---
            lse_start_time = time.time()
            # Use scaled y for LSE
            self._update_consequents_lse_ridge(X, y_scaled, rule_mus_norm)
            lse_time = time.time() - lse_start_time

            # --- 计算预测和误差 (基于 scaled y) ---
            X_plus = np.hstack([X, np.ones((X.shape[0], 1))])
            rule_outputs_scaled = X_plus @ self.consequents.T # These consequents predict scaled y
            y_pred_scaled = np.sum(rule_mus_norm * rule_outputs_scaled, axis=1).reshape(-1, 1)
            error_scaled = y_pred_scaled - y_scaled
            train_mse = np.mean(error_scaled**2) # MSE on scaled values
            history['train_loss'].append(train_mse)

            # --- 更新前件参数 (Gradient Descent) ---
            grad_start_time = time.time()
            grad_centers, grad_sigmas = self._backward_premise_gradient(
                X, y_pred_scaled.flatten(), y_scaled.flatten(), # Pass scaled y
                mus, rule_mus, rule_mus_norm, rule_outputs_scaled
            )
            self.centers -= lr_premise * grad_centers
            self.sigmas -= lr_premise * grad_sigmas
            self.sigmas = np.maximum(self.sigmas, 1e-6) # Ensure sigmas stay positive
            grad_time = time.time() - grad_start_time

            # --- 验证与早停 (基于 scaled y) ---
            val_mse = None
            if validation_data is not None:
                # Predict on validation set (returns scaled prediction)
                y_val_pred_scaled = self.predict(X_val, return_scaled=True).reshape(-1, 1)
                # Calculate MSE using scaled validation target
                val_mse = np.mean((y_val_pred_scaled - y_val_scaled)**2)
                history['val_loss'].append(val_mse)

                if early_stopping_patience is not None:
                    if val_mse < best_val_loss - 1e-6:
                        best_val_loss = val_mse
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                            break

            # --- 打印日志 ---
            epoch_time = time.time() - epoch_start_time
            log_interval = epochs // 20 if epochs >= 20 else 1
            if (epoch+1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
                # Report MSE based on scaled values for consistency during training
                log_msg = f"Epoch {epoch+1}/{epochs} | Train MSE (scaled): {train_mse:.6f}"
                if val_mse is not None:
                    log_msg += f" | Val MSE (scaled): {val_mse:.6f}"
                log_msg += f" | LSE Time: {lse_time:.3f}s | Grad Time: {grad_time:.3f}s | Total Epoch Time: {epoch_time:.3f}s"
                print(log_msg)

        print("--- Training Finished ---")
        return history

    # MODIFIED: Added option to return scaled or original prediction
    def predict(self, X, return_scaled=False):
        """
        使用训练好的模型进行预测。

        参数:
        - X: 输入数据
        - return_scaled: 如果为 True，返回标准化后的预测值；否则返回原始尺度的预测值。

        返回:
        - y_pred: 预测值 (scaled or original scale)
        """
        # Forward pass gives prediction in the scaled domain (if y was scaled)
        _, _, _, _, y_pred_internal = self._forward_pass_internal(X)
        y_pred_internal = y_pred_internal.reshape(-1, 1)

        if self._y_scaled and not return_scaled:
            # Inverse transform to get prediction in original scale
            y_pred = self.scaler_y.inverse_transform(y_pred_internal)
            return y_pred.flatten()
        else:
            # Return the internal prediction (which is scaled if y was scaled)
            return y_pred_internal.flatten()

    def get_rules(self, feature_names=None):
        """
        生成可读的模糊规则字符串。
        注意：后件参数是用于预测标准化后的 y 的。
        """
        if feature_names is None:
            feature_names = [f"x{i+1}" for i in range(self.n_input)]
        elif len(feature_names) != self.n_input:
             raise ValueError(f"Feature names length mismatch.")

        rules = []
        for rule_i, combo in enumerate(self._rule_indices):
            front_parts = [f"{feature_names[in_i]} IS {self.mf_names[in_i][mf_i]}"
                           for in_i, mf_i in enumerate(combo)]
            front_part = " AND ".join(front_parts)

            coeffs = self.consequents[rule_i]
            terms = [f"{coeffs[i]:+.4f}*{feature_names[i]}" for i in range(self.n_input)]
            const = f"{coeffs[-1]:+.4f}"
            # Add a note that the output is scaled
            then_part = "y_scaled = " + " ".join(terms) + f" {const}"

            rule_str = f"Rule {rule_i+1}: IF {front_part} THEN {then_part}"
            rules.append(rule_str)
        return rules

# --- Helper function for plotting (不变) ---
def plot_membership_functions(anfis_model, feature_index, feature_name, data_range=None, num_points=100):
    centers = anfis_model.centers[feature_index]
    sigmas = anfis_model.sigmas[feature_index]
    mfs_labels = anfis_model.mf_names[feature_index]
    if data_range:
        x_min, x_max = data_range
        padding = (x_max - x_min) * 0.1
        x = np.linspace(x_min - padding, x_max + padding, num_points)
        x_label_prefix = ""
    else:
        x = np.linspace(-3, 3, num_points)
        x_label_prefix = "Standardized "
    plt.figure(figsize=(8, 5))
    for i in range(anfis_model.n_mfs):
        mu = anfis_model.mf_func(x, centers[i], sigmas[i])
        plt.plot(x, mu, label=mfs_labels[i])
    plt.title(f"Input: {feature_name} (Index {feature_index}) Membership Functions")
    plt.xlabel(f"{x_label_prefix}{feature_name} Value")
    plt.ylabel("Membership Degree (μ)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



