import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def gaussian_mf(x, c, s):
    """
    高斯隶属函数
    x: 输入数组 (N,) 或单个值
    c: 高斯分布中心
    s: 标准差 (越大越平缓)
    返回: x 在该高斯模糊集中对应的隶属度
    """
    return np.exp(-0.5 * ((x - c) / (s + 1e-8))**2)

class SimpleANFIS:
    def __init__(self, n_input, n_mfs):
        """
        n_input: 输入维度 (例如 3 表示 三个特征)
        n_mfs: 每个输入的模糊隶属函数数量 (示例中为 2)

        本例中, 我们用高斯隶属函数, 每个输入维度 n_mfs 个 => 规则数 = (n_mfs ^ n_input)
        每条规则对应 1 个线性方程后件: y = p1*x1 + p2*x2 + p3*x3 + r
        """

        self.n_input = n_input
        self.n_mfs = n_mfs

        # 1) 隶属函数参数: centers, sigmas
        #    shape: (n_input, n_mfs) 每个输入维度有 n_mfs 个 (center, sigma)
        #    这里简单随机初始化，实际应用中可以通过聚类算法初始化
        self.centers = np.random.uniform(-1, 1, size=(n_input, n_mfs))
        self.sigmas  = np.random.uniform(0.5, 1.5, size=(n_input, n_mfs))

        # 2) 规则数 = n_mfs^n_input
        self.n_rules = (n_mfs ** n_input)

        # 3) 后件参数 (linear + constant)
        #    对于 3 个输入, 后件形式: y = p1*x1 + p2*x2 + p3*x3 + r
        #    对于 n_input 维, 后件: (p1, p2, ..., p_n, r), 共 n_input+1 个系数
        #    shape: (n_rules, n_input + 1)
        self.consequents = np.random.randn(self.n_rules, self.n_input + 1)

        # 为了方便打印规则, 给每个输入的 mfs 命个小名字:
        # 比如 "mf0", "mf1" ...
        self.mf_names = []
        for i in range(n_input):
            names = []
            for j in range(n_mfs):
                names.append(f"GaussianMF_{j}")
            self.mf_names.append(names)

    def forward(self, X):
        """
        前向传播:
        X: shape (N, n_input)
        返回:
          y_pred: shape (N,)
          rule_mus_norm: shape (N, n_rules) -> 每条规则的归一化激活度
        """
        N = X.shape[0]

        # 1) 计算每个输入在每个mf下的隶属度, shape (n_input, n_mfs, N)
        mus = []
        for i in range(self.n_input):
            # 对第 i 个输入:
            row_mus = []
            for j in range(self.n_mfs):
                c = self.centers[i, j]
                s = self.sigmas[i, j]
                mu_ij = gaussian_mf(X[:, i], c, s)  # shape (N,)
                row_mus.append(mu_ij)
            mus.append(row_mus)

        # 2) 计算规则激活度
        rule_indices = list(product(*[range(self.n_mfs) for _ in range(self.n_input)]))
        rule_mus = np.zeros((N, self.n_rules))

        for rule_i, combo in enumerate(rule_indices):
            # 计算每条规则的激活度: 乘积形式
            tmp = np.ones(N)
            for in_i, mf_i in enumerate(combo):
                tmp *= mus[in_i][mf_i]  # shape(N,)
            rule_mus[:, rule_i] = tmp

        # 3) 归一化
        rule_sum = np.sum(rule_mus, axis=1, keepdims=True) + 1e-8
        rule_mus_norm = rule_mus / rule_sum  # shape (N, n_rules)

        # 4) 计算后件输出
        #    对第 k 条规则, 后件 y_k = p1*x1 + p2*x2 + p3*x3 + r
        #    consequents[ rule_k ] shape (n_input+1,)
        ones = np.ones((N, 1))
        X_plus = np.hstack([X, ones])  # shape (N, n_input + 1)

        # 计算每条规则的后件输出
        # rule_out: shape (N, n_rules)
        rule_out = np.dot(X_plus, self.consequents.T)  # shape (N, n_rules)

        # 5) 计算最终输出
        y_pred = np.sum(rule_mus_norm * rule_out, axis=1)  # shape (N,)

        return y_pred, rule_mus_norm

    def backward(self, X, y, y_pred, rule_mus_norm, lr=0.01):
        """
        反向传播, 更新:
          1) 后件参数 self.consequents

        为简化示例, 仅对后件参数进行梯度下降更新
        """
        N = X.shape[0]

        # 1) 计算误差
        error = y_pred.reshape(-1, 1) - y  # shape (N,1)

        # 2) 更新后件参数
        # y_pred = sum_k (w_k * (p1_k * x1 + p2_k * x2 + p3_k * x3 + r_k))
        # dE/d(consequents[k]) = 2/N * sum_i (error_i * w_ik * x_i_plus)
        # where x_i_plus = [x1, x2, x3, 1]

        # 计算梯度
        # rule_mus_norm: (N, n_rules)
        # X_plus: (N, n_input +1)
        grad_consequents = (2.0 / N) * np.dot(rule_mus_norm.T, error * X)  # shape (n_rules, n_input)

        # 计算梯度对于偏置 r_k
        grad_consequents_r = (2.0 / N) * np.sum(rule_mus_norm * error, axis=0).reshape(-1,1)  # shape (n_rules,1)

        # 合并梯度
        grad_consequents_full = np.hstack([grad_consequents, grad_consequents_r])  # shape (n_rules, n_input +1)

        # 更新后件参数
        self.consequents -= lr * grad_consequents_full

    def fit(self, X, y, epochs=100, lr=0.01):
        """
        训练 ANFIS
        """
        for epoch in range(epochs):
            y_pred, rule_mus_norm = self.forward(X)
            self.backward(X, y, y_pred, rule_mus_norm, lr)

            if (epoch+1) % 10 == 0 or epoch == 0:
                mse = np.mean((y_pred - y)**2)
                print(f"Epoch {epoch+1}/{epochs}, MSE: {mse:.6f}")

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

    def get_rules(self):
        """
        打印/返回可读的模糊规则
        规则: IF x1 is MF? AND x2 is MF? AND x3 is MF? THEN y = p1*x1 + p2*x2 + p3*x3 + r
        """
        rule_indices = list(product(*[range(self.n_mfs) for _ in range(self.n_input)]))

        rules = []
        for rule_i, combo in enumerate(rule_indices):
            # 构造前件
            front_str_list = []
            for in_i, mf_i in enumerate(combo):
                front_str_list.append(f"x{in_i+1} IS {self.mf_names[in_i][mf_i]}")
            front_part = " AND ".join(front_str_list)

            # 后件: shape(n_input+1,) -> [p1, p2, p3, r]
            coeffs = self.consequents[rule_i]
            # 构造可读字符串
            terms = [f"{coeffs[i]:.4f}*x{i+1}" for i in range(self.n_input)]
            const = f"{coeffs[-1]:.4f}"
            then_part = "y = " + " + ".join(terms) + " + " + const

            rule_str = f"Rule {rule_i+1}: IF {front_part} THEN {then_part}"
            rules.append(rule_str)
        return rules

def plot_membership_functions(anfis_model, feature_index, feature_name, num_points=100):
        """
        绘制指定特征的所有高斯隶属函数

        参数：
        - anfis_model: 已训练的 SimpleANFIS 实例
        - feature_index: 特征的索引 (0, 1, 2, ...)
        - feature_name: 特征的名称 (字符串)
        - num_points: 绘图点的数量
        """
        centers = anfis_model.centers[feature_index]
        sigmas = anfis_model.sigmas[feature_index]
        mfs = [f"GaussianMF_{i}" for i in range(anfis_model.n_mfs)]

        x = np.linspace(-3, 3, num_points)  # 根据标准化后的数据范围调整
        plt.figure(figsize=(8, 6))
        for i in range(anfis_model.n_mfs):
            mu = gaussian_mf(x, centers[i], sigmas[i])
            plt.plot(x, mu, label=mfs[i])
        plt.title(f"Feature {feature_index + 1} ({feature_name}) Membership Functions")
        plt.xlabel(f"Standardized {feature_name}")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        plt.show()