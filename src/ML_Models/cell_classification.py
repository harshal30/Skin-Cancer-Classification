import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from numpy import mean, std
from matplotlib import pyplot as plt
import seaborn as sns 



def evaluate_model(X, y, model, n_splits=10, n_repeats=3, random_state=1):
    """
    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Targets.
        model: The ML model.
        n_splits (int): Number of folds in the cross-validation.
        n_repeats (int): Number of times cross-validator is repeated.
        random_state (int): Seed for reproducibility.

    Returns:
        numpy.ndarray: Array of AUC scores from each fold.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    return scores

def get_models():
    
    models, names = list(), list()

    # Logistic Regression
    models.append(LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42))
    names.append('LR')

    # Support Vector Machine
    models.append(SVC(gamma='scale', class_weight='balanced', probability=True, random_state=42))
    names.append('SVM')

    # Balanced Bagging Classifier 
    models.append(BalancedBaggingClassifier(n_estimators=1000, random_state=42))
    names.append('Bagging_DT') 

    # Balanced Random Forest Classifier
    models.append(BalancedRandomForestClassifier(n_estimators=1000, random_state=42))
    names.append('BRF') 

    # XGBoost Classifier
    # 
    models.append(XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1.01, random_state=42))
    names.append('XGBoost')

    return models, names

def run_classification(data_path, output_plot_path=None):
    """
    Args:
        data_path (str): Path to the combined CSV file containing features and labels.
        output_plot_path (str, optional): Path to save the box plot of AUC scores.
                                         If None, the plot will be displayed.
    """
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)

    target = data['label']
    original_features = data.drop('label', axis=1)

    print(f"Original target distribution:\n{target.value_counts()}")

    target = target.map({'HDF': 1, 'A375': 0})

    # Apply ADASYN oversampling to handle class imbalance
    print("Applying ADASYN oversampling...")
    oversample = ADASYN(random_state=42)
    original_features, target = oversample.fit_resample(original_features, target)
    print(f"Target distribution after ADASYN:\n{target.value_counts()}")

    print("Performing feature selection...")
    standard_features = (original_features - original_features.mean()) / original_features.std()
    X_train, X_test, Y_train, Y_test = train_test_split(standard_features, target, test_size=0.1, random_state=42)

    rf_selector = BalancedRandomForestClassifier(n_estimators=1000, random_state=42)
    # Use a pipeline for feature selection as well, including PowerTransformer
    steps_selector = [('p', PowerTransformer()), ('m', rf_selector)]
    pipeline_selector = Pipeline(steps=steps_selector)
    pipeline_selector.fit(X_train, Y_train)

    importances = rf_selector.feature_importances_
    std_importances = np.std([tree.feature_importances_ for tree in rf_selector.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    mean_std_importance = std_importances.mean()
    selected_feature_names = [original_features.columns[i] for i in range(len(importances)) if importances[i] > mean_std_importance]
    new_features = original_features[selected_feature_names]

    print(f"Selected {len(selected_feature_names)} features out of {original_features.shape[1]}:")
    print(selected_feature_names)

    # Plot feature importances
    plt.figure(figsize=(15, 8))
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center", yerr=std_importances[indices])
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Importance", fontsize=14)
    plt.title("Feature Importances", fontsize=16)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()

    print("\nEvaluating models...")
    models, names = get_models()
    results = list()

    for i in range(len(models)):
        # Define pipeline steps for each model: PowerTransformer + Model
        steps = [('p', PowerTransformer()), ('m', models[i])]
        pipeline = Pipeline(steps=steps)

        # Evaluate the pipeline and store results
        scores = evaluate_model(new_features, target, pipeline)
        results.append(scores)

        print('>%s AUC: %.3f (%.3f)' % (names[i], mean(scores), std(scores)))

    plt.figure(figsize=(10, 7))
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.xlabel('ML Algorithms', fontsize=15)
    pyplot.ylabel('AUC Score', fontsize=15)
    plt.ylim([0.83, 1])
    pyplot.xticks(fontsize=12)
    pyplot.yticks(fontsize=12)
    pyplot.grid(False)
    pyplot.title("Model Performance (AUC Score)", fontsize=16)
    pyplot.tight_layout()

    if output_plot_path:
        plt.savefig(output_plot_path)
        print(f"Saved model performance plot to {output_plot_path}")
    else:
        pyplot.show()

if __name__ == "__main__":
    combined_data_path = "D:/Skin Cancer/dataset/Properties/combined_mid.csv"
    output_plot_file = "model_performance_auc.png"

    run_classification(combined_data_path, output_plot_file)

