# src/feature_engineer.py

import pandas as pd
import numpy as np

class FeatureEngineer:
    """Handles all feature engineering tasks for the ICR competition."""

    def __init__(self):
        """Initializes the feature engineer."""
        self.learned_stats = None

    def fit(self, df):
        """
        Learns necessary statistics from the training data for feature creation.
        
        Args:
            df (pd.DataFrame): The training dataframe.
        """
        print("Step 2: Learning feature parameters from training data...")
        # In a real-world scenario, you might calculate means, std devs, etc.
        # For this specific notebook, the feature engineering was direct.
        # We will keep this structure for good practice.
        self.learned_stats = "learned" # Placeholder

    def transform(self, df):
        """
        Applies feature transformations to the data.
        
        Args:
            df (pd.DataFrame): The dataframe to transform (train or test).
            
        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        if self.learned_stats is None:
            raise RuntimeError("You must call 'fit' before 'transform'.")
        
        print("Applying feature transformations...")
        
        # Handle 'EJ' categorical feature by converting it to a number
        df['EJ_Cat'] = df['EJ'].replace({'A': 1, 'B': 2})
        
        # This section is kept simple to match your original notebook's logic.
        # More complex features based on learned stats would go here.
        # For example, creating interaction features.
        
        # A simple interaction feature example (can be expanded)
        # Note: Your original notebook had more complex interactions. This is a template.
        for col in ['AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD ', 'BN', 'BP', 'BQ', 'BR', 'BT', 'BV', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL', 'CR', 'CS', 'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY', 'EB', 'EE', 'EG', 'EH', 'EL', 'EP', 'EU', 'FC', 'FD ', 'FE', 'FI', 'FL', 'FR', 'FS', 'GB', 'GE', 'GF', 'GH', 'GI', 'GL']:
             if col in df.columns:
                df[f'{col}_sq'] = df[col] ** 2


        return df

    def fit_transform(self, df):
        """Convenience method to fit on and then transform the same data."""
        self.fit(df)
        return self.transform(df)
