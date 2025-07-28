Skin Cancer Cell Classification
This repository contains Python code for classifying skin cancer and normal cells based on morphological and phase based features extracted from phase images. The pipeline involves feature extraction, data preprocessing (including handling class imbalance), feature selection, and evaluation of various machine learning models.

<!-- Project Structure
skin_cancer_classification/
├── src/
│   ├── data_processing/
│   │   └── feature_extraction.py   # Functions for extracting cell properties
│   ├── models/
│   │   └── cell_classifier.py      # ML model training and evaluation pipeline
│   └── utils/
│       └── metrics.py              # Utility functions and custom metrics (placeholders for now)
├── data/
│   ├── raw/                        # Directory for raw mask (.npy) and OPD (.mat) files
│   │   ├── masks/
│   │   └── opd_values/
│   └── processed/                  # Directory for processed data (e.g., combined_mid.csv)
│       └── properties/
├── .gitignore                      # Specifies intentionally untracked files to ignore
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── LICENSE                         # Project license (e.g., MIT, Apache 2.0) -->

Install dependencies:

pip install -r requirements.txt

Data
The project expects raw mask files (.npy) and optical path difference (OPD) files (.mat).

data/raw/masks/: Place your cell mask .npy files here.

data/raw/opd_values/: Place your OPD .mat files here.

Usage
1. Feature Extraction
Run the feature_extraction.py script to extract morphological and phase based features from your raw data. This will generate individual CSV files for each mask/OPD pair.

python src/data_processing/feature_extraction.py

After running this, you will need to combine the individual CSVs generated in data/processed/properties/ into a single combined_mid.csv file for both cell types. This combined_mid.csv should also include a label column ('HDF' or 'A375') for classification.

2. Cell Classification
Once data/processed/properties/combined_mid.csv is ready with the extracted features and labels, you can run the cell_classifier.py script to train and evaluate the classification models.

python src/models/cell_classifier.py

This script will perform:

Data loading and label mapping.

ADASYN oversampling for class imbalance.

Feature selection using BalancedRandomForestClassifier and Information gain.

Evaluation of Logistic Regression, SVM, Balanced Bagging Classifier, Balanced Random Forest, and XGBoost using Repeated Stratified K-Fold cross-validation.

It will print the mean AUC score and standard deviation for each model and display a box plot of the AUC scores.

Models Evaluated
The following machine learning algorithms are evaluated:

Logistic Regression (LR)

Support Vector Machine (SVM)

Balanced Bagging Classifier (Bagging_DT)

Balanced Random Forest Classifier (BRF)

XGBoost Classifier (XGBoost)

All models are integrated into a pipeline with PowerTransformer for feature scaling.
