
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal

class FeatureSelector:
    def __init__(self):
        self.selected_features = []

    def correlation_filter(self, df: pd.DataFrame, threshold: float = 0.65):
        corr = df.corr().abs()
        upper = corr.where(~(corr < threshold))
        return upper.dropna(axis=1, how='all').columns.tolist()

    def statistical_tests(self, df: pd.DataFrame, y: pd.Series):
        numerical_features = df.select_dtypes(include='number').columns
        significant = []
        for feature in numerical_features:
            groups = [df[feature][y == cls] for cls in y.unique()]
            try:
                if len(groups) == 2:
                    stat, p = mannwhitneyu(groups[0], groups[1])
                elif len(groups) > 2:
                    stat, p = kruskal(*groups)
                else:
                    continue
                if p < 0.05:
                    significant.append(feature)
            except Exception:
                continue
        return significant
