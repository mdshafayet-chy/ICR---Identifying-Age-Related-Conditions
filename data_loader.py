# src/data_loader.py

import pandas as pd
import os

class DataLoader:
    """Handles loading and the initial merging of the competition data."""

    def __init__(self, data_dir):
        """
        Initializes the loader with the path to the data directory.
        
        Args:
            data_dir (str): The path to the directory containing the CSV files.
        """
        self.data_dir = data_dir

    def load_data(self):
        """
        Loads the train, test, and greeks CSV files and performs the initial merge.
        
        Returns:
            tuple: A tuple containing the merged training DataFrame and testing DataFrame.
        """
        print("Step 1: Loading data...")
        try:
            train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
            greeks_df = pd.read_csv(os.path.join(self.data_dir, 'greeks.csv'))
        except FileNotFoundError as e:
            print(f"Error: {e}. Make sure your data files are in the '{self.data_dir}' directory.")
            return None, None
