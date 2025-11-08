import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

class HigeeAnn:
    def __init__(self, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
        # 接收预处理后的数据张量
        self.X_train_tensor = X_train_tensor
        self.y_train_tensor = y_train_tensor
        self.X_test_tensor = X_test_tensor
        self.y_test_tensor = y_test_tensor

    def train_model(self, model, epochs=1000, lr=0.001):
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 创建数据加载器
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # 训练模型
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')
        return model

    def evaluate_model(self, model):
        model.eval()
        with torch.no_grad():
            predictions = model(self.X_test_tensor).squeeze().numpy()
            mse = np.mean((predictions - self.y_test_tensor.numpy()) ** 2)
            print(f'Mean Squared Error on Test Set: {mse:.4f}')
        return predictions

    def ann(self):
        # 定义 ANN 模型
        class ANNModel(nn.Module):
            def __init__(self, input_size):
                super(ANNModel, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = ANNModel(self.X_train_tensor.shape[1])
        model = self.train_model(model)
        predictions = self.evaluate_model(model)
        return model, predictions

    def cnn(self):
        # 为 CNN 调整输入数据形状
        X_train_tensor_cnn = self.X_train_tensor.unsqueeze(1)  # 增加通道维度
        X_test_tensor_cnn = self.X_test_tensor.unsqueeze(1)

        # 定义 CNN 模型
        class CNNModel(nn.Module):
            def __init__(self, input_size):
                super(CNNModel, self).__init__()
                self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
                self.fc1 = nn.Linear(32 * input_size, 64)
                self.fc2 = nn.Linear(64, 1)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)  # Flatten
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = CNNModel(self.X_train_tensor.shape[1])
        model = self.train_model(model)
        predictions = self.evaluate_model(model)
        return model, predictions

    def lstm(self):
        # 为 LSTM 调整输入数据形状
        X_train_tensor_lstm = self.X_train_tensor.unsqueeze(1)  # 增加时间步维度
        X_test_tensor_lstm = self.X_test_tensor.unsqueeze(1)

        # 定义 LSTM 模型
        class LSTMModel(nn.Module):
            def __init__(self, input_size):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size=input_size, hidden_size=50, batch_first=True)
                self.fc1 = nn.Linear(50, 1)

            def forward(self, x):
                _, (hn, _) = self.lstm(x)
                x = hn[-1]  # 取最后一个时间步的隐藏状态
                x = self.fc1(x)
                return x

        model = LSTMModel(self.X_train_tensor.shape[1])
        model = self.train_model(model)
        predictions = self.evaluate_model(model)
        return model, predictions



# 加载数据
file_path = 'H2S_latest.xlsx'
data = pd.read_excel(file_path)

# 特征和目标变量
features = ['rpm', 'l_flow', 'ph', 'tannic_conc', 'sodium_conc']
target = 'y'

# 提取特征和目标
X = data[features].values
y = data[target].values

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 使用处理后的数据创建模型实例
higee_model = HigeeAnn(X_train, y_train, X_test, y_test)

# 训练 ANN 模型
ann_model, ann_predictions = higee_model.ann()

# 训练 CNN 模型
# cnn_model, cnn_predictions = higee_model.cnn()
#
# # 训练 LSTM 模型
# lstm_model, lstm_predictions = higee_model.lstm()
