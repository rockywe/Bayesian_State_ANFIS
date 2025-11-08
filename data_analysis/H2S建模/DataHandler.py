import pandas as pd

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_excel(self.file_path)
        print("数据加载完成:")
        print(self.data.head())