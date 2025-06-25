import pandas as pd

class DataLoader:
    def __init__(self, train_path: str, greeks_path: str, test_path: str):
        self.train_path = train_path
        self.greeks_path = greeks_path
        self.test_path = test_path

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        greeks_df = pd.read_csv(self.greeks_path)
        test_df = pd.read_csv(self.test_path)
        return train_df, greeks_df, test_df
