import os
import pandas as pd
import pytest
from src.preprocessing import get_preprocessor

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")

def test_data_files_exist():
    assert os.path.exists(os.path.join(DATA_DIR, "train.csv")), "train.csv not found in data/processed"
    assert os.path.exists(os.path.join(DATA_DIR, "test.csv")), "test.csv not found in data/processed"

def test_data_shapes():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    assert not train_df.empty, "train.csv is empty"
    assert not test_df.empty, "test.csv is empty"
    assert "target" in train_df.columns, "target column missing in train.csv"
    assert "target" in test_df.columns, "target column missing in test.csv"
    assert train_df.shape[1] == 14, "train.csv must have 14 columns"
    assert test_df.shape[1] == 14, "test.csv must have 14 columns"

def test_preprocessing_pipeline():
    dummy_data = pd.DataFrame([{
        "age": 52.0, "sex": 1.0, "cp": 3.0, "trestbps": 125.0, "chol": 212.0,
        "fbs": 0.0, "restecg": 1.0, "thalach": 168.0, "exang": 0.0, "oldpeak": 1.0,
        "slope": 1.0, "ca": 2.0, "thal": 3.0
    }])
    
    preprocessor = get_preprocessor()
    preprocessor.fit(dummy_data)
    transformed = preprocessor.transform(dummy_data)
    
    assert transformed.shape[0] == 1, "Transformed shape must have 1 row"
    assert transformed.shape[1] > 0, "Transformed feature dimension must be non-zero"
