
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_selector import FeatureSelector
from src.model import XGBoostModel
from src.evaluator import Evaluator

class Pipeline:
    def __init__(self):
        self.loader = DataLoader('data/train.csv', 'data/greeks.csv', 'data/test.csv')
        self.preprocessor = Preprocessor()
        self.selector = FeatureSelector()
        self.model = XGBoostModel()

    def run(self):
        train_df, _, test_df = self.loader.load_data()

        imputed_train = self.preprocessor.impute(train_df)
        encoded_train = self.preprocessor.encode_ej(train_df)
        full_train = pd.concat([imputed_train, encoded_train[['EJ', 'Class']]], axis=1)

        y = full_train['Class']
        X = full_train.drop(columns=['Class'])

        selected_corr = self.selector.correlation_filter(X)
        selected_stat = self.selector.statistical_tests(X[selected_corr], y)

        final_X = X[selected_stat]
        X_train, X_test, y_train, y_test = train_test_split(final_X, y, test_size=0.2, stratify=y, random_state=42)

        self.model.train(X_train, y_train)
        preds = self.model.predict(X_test)

        evaluator = Evaluator(y_test, preds)
        evaluator.print_metrics()
        
        evaluator.plot_confusion_matrix()

        # Make test predictions and create submission
        imputed_test = self.preprocessor.impute(test_df)
        encoded_test = self.preprocessor.encode_ej(test_df)
        full_test = pd.concat([imputed_test, encoded_test[['EJ']]], axis=1)
        test_final = full_test[selected_stat]

        test_preds_proba = self.model.predict_proba(test_final)
        submission = pd.DataFrame({'Id': test_df['Id'], 'Class': test_preds_proba})
        submission.to_csv('submission.csv', index=False)
        print("Submission file saved as 'submission.csv'")
 
 