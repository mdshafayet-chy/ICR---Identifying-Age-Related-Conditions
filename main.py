# main.py

import pandas as pd
import numpy as np

# Import our custom classes from the 'src' package
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer

# --- Configuration ---
# This section makes it easy to change settings without digging into the code.
DATA_DIR = 'data'  # IMPORTANT: Your data must be in this folder
OUTPUT_FILE = 'submission.csv'
MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.04,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

def main():
    """Main function to execute the full ML pipeline."""
    
    print("--- Starting ICR Prediction Pipeline ---")
    
    # 1. Load Data using the DataLoader class
    data_loader = DataLoader(data_dir=DATA_DIR)
    train_df, test_df = data_loader.load_data()
    
    if train_df is None:
        print("Halting execution due to data loading error.")
        return

    # Store test IDs for the final submission file
    test_ids = test_df['Id'].copy()
    
    # 2. Engineer Features using the FeatureEngineer class
    feature_engineer = FeatureEngineer()
    # Fit on the training data and transform it
    train_featured_df = feature_engineer.fit_transform(train_df)
    # Use the same learned parameters to transform the test data
    test_featured_df = feature_engineer.transform(test_df)
    
    # 3. Prepare Data for the Model
    # Define features to use (all columns except identifiers and the target)
    features_to_exclude = ['Id', 'Class', 'Alpha', 'EJ']
    features = [col for col in train_featured_df.columns if col not in features_to_exclude]
    
    X_train = train_featured_df[features]
    y_train = train_featured_df['Class']
    X_test = test_featured_df[features]
    
    # 4. Train Model using the ModelTrainer class
    model_trainer = ModelTrainer(model_params=MODEL_PARAMS)
    model_trainer.train(X_train, y_train)
    
    # 5. Get Predictions
    predictions = model_trainer.predict(X_test)
    
    # 6. Generate Submission File
    print(f"Step 5: Generating submission file: {OUTPUT_FILE}")
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'class_0': predictions[:, 0],
        'class_1': predictions[:, 1]
    })
    submission_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"--- Pipeline Finished Successfully! ---")
    print(f"Submission file saved as '{OUTPUT_FILE}'.")

# This standard Python construct makes the script executable
if __name__ == '__main__':
    main()
