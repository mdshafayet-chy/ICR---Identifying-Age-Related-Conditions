
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def print_metrics(self):
        print(f"Accuracy: {accuracy_score(self.y_true, self.y_pred):.4f}")
        print(f"F1 Score: {f1_score(self.y_true, self.y_pred):.4f}")

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()