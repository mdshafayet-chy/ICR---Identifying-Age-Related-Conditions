# ICR - Identifying Age-Related Conditions (Kaggle Competition)

> 🏆 **Silver Medal Solution** — Ranked **83rd out of 8,430 teams**

This repository contains the complete codebase for the Silver Medal-winning solution for the **"ICR - Identifying Age-Related Conditions"** Kaggle competition. The project was developed by **Md Shafayet Hossen Chowdhury**.

---

##  Project Goal

The objective of this competition was to build a machine learning model that could predict whether an individual shows signs of one or more **age-related conditions (Class 1)** or **none (Class 0)**, based on anonymized health measurements.

This repository showcases a clean, modular, and production-ready machine learning pipeline that yielded a top-tier result on the leaderboard.

---

##  Features

- 📥 Clean data loading and preprocessing pipeline
- 🧹 KNN imputation and label encoding
- 🔬 Feature selection using:
  - Correlation filtering
  - Mann-Whitney U Test
  - Kruskal-Wallis H Test
- 📈 Model training with XGBoost
- 🧪 Evaluation with accuracy, F1 score & confusion matrix
- 📦 Test-time prediction with submission file generator
- 💾 Model persistence with `joblib`

---

## Project Structure
```bash
project_root/
│
├── main.py                     # 🚀 Entry point to run the full ML pipeline
├── submission.csv              # 📝 Final prediction CSV (Kaggle submission format)
├── environment.yml             # 📦 Conda environment file (dependencies)
├── .gitignore                  # 🚫 Specifies untracked files for Git
│
├── data/                       # 📊 Input data files
│   ├── train.csv               # 🔧 Training dataset (includes features and Class)
│   ├── test.csv                # 🧪 Test dataset (used for submission predictions)
│   └── greeks.csv              # 📄 Metadata (ID + class label timing)
│
└── src/                        # 🧠 Modular components for the pipeline
    ├── data_loader.py          # 📥 Loads train, test, and greeks CSVs
    ├── preprocessor.py         # 🧹 Handles KNN imputation and EJ label encoding
    ├── feature_selector.py     # 🔍 Selects relevant features using correlation and statistical tests
    ├── model.py                # 🤖 Defines and manages the XGBoost model (train, predict, save, load)
    ├── evaluator.py            # 📊 Computes accuracy, F1, and shows confusion matrix
    └── pipeline.py             # 🧩 Connects all modules into a complete workflow
```


-----
##  Setup

>  Recommended: Create and activate the environment using the provided `.yml` file.

```bash
conda env create -f environment.yml
conda activate xgboost-pipeline
```
##How to Run
Make sure the dataset files are in the data/ directory as shown below.

Then, execute the pipeline:

```bash

python main.py

```
You’ll see:

Accuracy and F1-score printed to the console

Confusion matrix visualized

A trained model saved to models/xgb_model.pkl```text

All required dependencies are listed in the environment.yml file. Notable ones include:
```
xgboost
scikit-learn
pandas
seaborn
matplotlib
numpy
```

## License
This project is shared for educational purposes. Use responsibly and credit the original author if reused in public repositories or solutions.

