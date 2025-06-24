ICR - Identifying Age-Related Conditions (Kaggle Competition)
🏆 Silver Medal Solution 🏆
This repository contains the code for the Silver Medal-winning solution (83rd place out of 8,430 teams) for the "ICR - Identifying Age-Related Conditions" Kaggle competition. The project was developed by MD Shafayet Hossen Chowdhury.

Project Goal
The goal of this competition was to build a machine learning model to predict the presence of one or more of three medical conditions (Class 1) or the absence of all three (Class 0), based on anonymous health measurements. This project demonstrates a complete, modular pipeline for achieving a top-ranking result.

🚀 Getting Started
This project is structured as a modular Python application. Follow the steps below to set up and run the pipeline.

Prerequisites
Python 3.8+

pip for package management

1. Project Structure
The repository is organized as follows:

icr_competition/
|-- data/
|   |-- train.csv
|   |-- test.csv
|   `-- greeks.csv
|
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- feature_engineer.py
|   `-- model_trainer.py
|
|-- main.py
`-- README.md

2. Installation
Clone the repository (or set up the files as provided):

git clone [your-repo-url]
cd icr_competition

Create a requirements.txt file with the following content:

pandas
numpy
xgboost
scikit-learn

Install the required packages:

pip install -r requirements.txt

Place the data: Download the competition data from Kaggle and place train.csv, test.csv, and greeks.csv into the data/ directory.

3. How to Run
To execute the full pipeline—from loading data to generating the final submission.csv file—run the main.py script from the root directory:

python main.py

The script will print status updates for each step and will create the submission.csv file in the project's root directory upon successful completion.

🔧 Code Architecture
The pipeline is broken down into distinct classes, each with a single responsibility.

src/data_loader.py
Class: DataLoader

Responsibility: Loads the three .csv files (train, test, greeks) and performs the initial merge to add the Alpha feature to the main datasets.

src/feature_engineer.py
Class: FeatureEngineer

Responsibility: Handles all data transformations and feature creation. It learns any necessary parameters from the training set (fit method) and applies them consistently to both the training and test sets (transform method).

src/model_trainer.py
Class: ModelTrainer

Responsibility: Manages the XGBoost model. It handles the model's initialization, training on the prepared data, and making predictions on the test set.

main.py
Orchestrator Script: This is the entry point of the application. It initializes and calls the methods of the above classes in the correct sequence to run the entire pipeline. It also contains the configuration settings (like model parameters and file paths) for easy modification.
