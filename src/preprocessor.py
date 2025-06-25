
import pandas as pd
from sklearn.impute import KNNImputer

class Preprocessor:
    def __init__(self):
        self.imputer = KNNImputer()

    def impute(self, df: pd.DataFrame, drop_cols=['Id', 'EJ', 'Class']):
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        imputed_data = self.imputer.fit_transform(X)
        imputed_df = pd.DataFrame(imputed_data, columns=X.columns)
        return imputed_df

    def encode_ej(self, df: pd.DataFrame):
        if 'EJ' in df.columns:
            df['EJ'] = df['EJ'].map({'A': 0, 'B': 1})
        return df