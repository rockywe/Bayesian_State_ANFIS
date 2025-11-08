import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import pinv


class FuzzyModel:
    def __init__(self, pH, rpm, ch2o2, error):
        self.pH = pH
        self.rpm = rpm
        self.ch2o2 = ch2o2
        self.error = error

        # 初始化模型的隶属函数
        self.y1 = self.trimf(self.pH, [11.3, 11.8, 12.45])
        self.y2 = self.trimf(self.pH, [12.3, 12.9, 13.7])
        self.y3 = self.trimf(self.rpm, [190, 495, 760])
        self.y4 = self.trimf(self.rpm, [750, 1320, 1700])
        self.y5 = self.trimf(self.ch2o2, [0, 0.2, 0.5])
        self.y6 = self.trimf(self.ch2o2, [0.4, 0.65, 0.85])
        self.y7 = self.trimf(self.ch2o2, [0.8, 1.2, 1.70])

        # 计算模糊规则隶属度
        self.calculate_membership()

        # 构建大矩阵
        self.build_design_matrix()

        # 计算P矩阵
        self.P = self.calculate_P_matrix()

    def trimf(self, x, abc):
        a, b, c = abc
        return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

    def calculate_membership(self):
        self.mem1 = self.y1 * self.y3 * self.y5
        self.mem2 = self.y1 * self.y3 * self.y6
        self.mem3 = self.y1 * self.y3 * self.y7
        self.mem4 = self.y1 * self.y4 * self.y5
        self.mem5 = self.y1 * self.y4 * self.y6
        self.mem6 = self.y1 * self.y4 * self.y7
        self.mem7 = self.y2 * self.y3 * self.y5
        self.mem8 = self.y2 * self.y3 * self.y6
        self.mem9 = self.y2 * self.y3 * self.y7
        self.mem10 = self.y2 * self.y4 * self.y5
        self.mem11 = self.y2 * self.y4 * self.y6
        self.mem12 = self.y2 * self.y4 * self.y7

    def build_design_matrix(self):
        denom = (self.mem1 + self.mem2 + self.mem3 + self.mem4 + self.mem5 + self.mem6 +
                 self.mem7 + self.mem8 + self.mem9 + self.mem10 + self.mem11 + self.mem12)

        beta1 = self.mem1 / denom
        beta2 = self.mem2 / denom
        beta3 = self.mem3 / denom
        beta4 = self.mem4 / denom
        beta5 = self.mem5 / denom
        beta6 = self.mem6 / denom
        beta7 = self.mem7 / denom
        beta8 = self.mem8 / denom
        beta9 = self.mem9 / denom
        beta10 = self.mem10 / denom
        beta11 = self.mem11 / denom
        beta12 = self.mem12 / denom

        # 构建大矩阵Z
        self.Z = np.zeros((len(self.pH), 48))

        # 填充Z矩阵
        self.Z[:, 0] = beta1
        self.Z[:, 1] = beta2
        self.Z[:, 2] = beta3
        self.Z[:, 3] = beta4
        self.Z[:, 4] = beta5
        self.Z[:, 5] = beta6
        self.Z[:, 6] = beta7
        self.Z[:, 7] = beta8
        self.Z[:, 8] = beta9
        self.Z[:, 9] = beta10
        self.Z[:, 10] = beta11
        self.Z[:, 11] = beta12

        # pH相关的beta值
        self.Z[:, 12] = beta1 * self.pH
        self.Z[:, 13] = beta2 * self.pH
        self.Z[:, 14] = beta3 * self.pH
        self.Z[:, 15] = beta4 * self.pH
        self.Z[:, 16] = beta5 * self.pH
        self.Z[:, 17] = beta6 * self.pH
        self.Z[:, 18] = beta7 * self.pH
        self.Z[:, 19] = beta8 * self.pH
        self.Z[:, 20] = beta9 * self.pH
        self.Z[:, 21] = beta10 * self.pH
        self.Z[:, 22] = beta11 * self.pH
        self.Z[:, 23] = beta12 * self.pH

        # 转速相关的beta值
        self.Z[:, 24] = beta1 * self.rpm
        self.Z[:, 25] = beta2 * self.rpm
        self.Z[:, 26] = beta3 * self.rpm
        self.Z[:, 27] = beta4 * self.rpm
        self.Z[:, 28] = beta5 * self.rpm
        self.Z[:, 29] = beta6 * self.rpm
        self.Z[:, 30] = beta7 * self.rpm
        self.Z[:, 31] = beta8 * self.rpm
        self.Z[:, 32] = beta9 * self.rpm
        self.Z[:, 33] = beta10 * self.rpm
        self.Z[:, 34] = beta11 * self.rpm
        self.Z[:, 35] = beta12 * self.rpm

        # H2O2相关的beta值
        self.Z[:, 36] = beta1 * self.ch2o2
        self.Z[:, 37] = beta2 * self.ch2o2
        self.Z[:, 38] = beta3 * self.ch2o2
        self.Z[:, 39] = beta4 * self.ch2o2
        self.Z[:, 40] = beta5 * self.ch2o2
        self.Z[:, 41] = beta6 * self.ch2o2
        self.Z[:, 42] = beta7 * self.ch2o2
        self.Z[:, 43] = beta8 * self.ch2o2
        self.Z[:, 44] = beta9 * self.ch2o2
        self.Z[:, 45] = beta10 * self.ch2o2
        self.Z[:, 46] = beta11 * self.ch2o2
        self.Z[:, 47] = beta12 * self.ch2o2

    def calculate_P_matrix(self):
        return pinv(self.Z.T @ self.Z) @ self.Z.T @ self.error

    def predict(self, ph, rpm, h2o2):
        # 模糊模型预测
        y11 = self.trimf(ph, [11.3, 11.8, 12.45])
        y22 = self.trimf(ph, [12.3, 12.9, 13.7])

        y33 = self.trimf(rpm, [190, 495, 760])
        y44 = self.trimf(rpm, [750, 1320, 1700])

        y55 = self.trimf(h2o2, [0, 0.2, 0.5])
        y66 = self.trimf(h2o2, [0.4, 0.65, 0.85])
        y77 = self.trimf(h2o2, [0.8, 1.2, 1.70])

        mem11 = y11 * y33 * y55
        mem22 = y11 * y33 * y66
        mem33 = y11 * y33 * y77
        mem44 = y11 * y44 * y55
        mem55 = y11 * y44 * y66
        mem66 = y11 * y44 * y77

        mem77 = y22 * y33 * y55
        mem88 = y22 * y33 * y66
        mem99 = y22 * y33 * y77
        mem101 = y22 * y44 * y55
        mem111 = y22 * y44 * y66
        mem121 = y22 * y44 * y77

        fuzzy_output = (mem11 * (self.P[0] + self.P[12] * ph + self.P[24] * rpm + self.P[36] * h2o2) +
                        mem22 * (self.P[1] + self.P[13] * ph + self.P[25] * rpm + self.P[37] * h2o2) +
                        mem33 * (self.P[2] + self.P[14] * ph + self.P[26] * rpm + self.P[38] * h2o2) +
                        mem44 * (self.P[3] + self.P[15] * ph + self.P[27] * rpm + self.P[39] * h2o2) +
                        mem55 * (self.P[4] + self.P[16] * ph + self.P[28] * rpm + self.P[40] * h2o2) +
                        mem66 * (self.P[5] + self.P[17] * ph + self.P[29] * rpm + self.P[41] * h2o2) +
                        mem77 * (self.P[6] + self.P[18] * ph + self.P[30] * rpm + self.P[42] * h2o2) +
                        mem88 * (self.P[7] + self.P[19] * ph + self.P[31] * rpm + self.P[43] * h2o2) +
                        mem99 * (self.P[8] + self.P[20] * ph + self.P[32] * rpm + self.P[44] * h2o2) +
                        mem101 * (self.P[9] + self.P[21] * ph + self.P[33] * rpm + self.P[45] * h2o2) +
                        mem111 * (self.P[10] + self.P[22] * ph + self.P[34] * rpm + self.P[46] * h2o2) +
                        mem121 * (self.P[11] + self.P[23] * ph + self.P[35] * rpm + self.P[47] * h2o2)) / \
                       (
                                   mem11 + mem22 + mem33 + mem44 + mem55 + mem66 + mem77 + mem88 + mem99 + mem101 + mem111 + mem121)

        return fuzzy_output

