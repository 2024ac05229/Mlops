import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay
)
from preprocessing import get_preprocessor

# Define paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load datasets
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Separate features and target
X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# Set MLflow experiment
mlflow.set_experiment("Heart_Disease_Classifier_Experiment")

def train_and_evaluate(model_name, base_clf, param_grid):
    print(f"\n--- Training {model_name} ---")
    
    # Create the complete pipeline: Preprocessor + Classifier
    preprocessor = get_preprocessor()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", base_clf)
    ])
    
    # Grid search cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=model_name) as run:
        print("Running grid search hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV ROC-AUC: {best_score:.4f}")
        
        # Log best parameters to MLflow
        for param_name, param_val in best_params.items():
            mlflow.log_param(param_name, param_val)
        mlflow.log_metric("cv_best_roc_auc", best_score)
        
        # Predict on test set
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate test metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-score: {test_f1:.4f}")
        print(f"Test ROC-AUC: {test_roc_auc:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        
        # Plot and log confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, 
            y_pred, 
            display_labels=["No Disease", "Disease"], 
            cmap=plt.cm.Blues, 
            ax=ax_cm
        )
        plt.title(f"Confusion Matrix - {model_name}")
        cm_plot_path = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(cm_plot_path)
        plt.close()
        mlflow.log_artifact(cm_plot_path)
        os.remove(cm_plot_path)
        
        # Plot and log ROC curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        RocCurveDisplay.from_predictions(
            y_test, 
            y_prob, 
            name=model_name, 
            ax=ax_roc
        )
        plt.title(f"ROC Curve - {model_name}")
        roc_plot_path = f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(roc_plot_path)
        plt.close()
        mlflow.log_artifact(roc_plot_path)
        os.remove(roc_plot_path)
        
        # Log the model pipeline
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model", serialization_format="pickle")
        
        return best_pipeline, test_roc_auc

if __name__ == "__main__":
    # Hyperparameter grids
    lr_params = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"]
    }
    
    rf_params = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5]
    }
    
    # Train Logistic Regression
    lr_pipeline, lr_auc = train_and_evaluate(
        "Logistic Regression",
        LogisticRegression(random_state=42, max_iter=1000),
        lr_params
    )
    
    # Train Random Forest
    rf_pipeline, rf_auc = train_and_evaluate(
        "Random Forest",
        RandomForestClassifier(random_state=42),
        rf_params
    )
    
    # Save the best model locally
    best_pipeline = rf_pipeline if rf_auc >= lr_auc else lr_pipeline
    best_name = "Random Forest" if rf_auc >= lr_auc else "Logistic Regression"
    print(f"\nBest Model selected: {best_name} with ROC-AUC: {max(rf_auc, lr_auc):.4f}")
    
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"Saved the best pipeline to {best_model_path}")
