import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Create directories if they don't exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Dataset details
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
RAW_FILE_PATH = os.path.join(RAW_DIR, "heart_disease_raw.csv")
TRAIN_FILE_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(PROCESSED_DIR, "test.csv")

# Columns of UCI heart disease Cleveland dataset
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def download_data():
    print(f"Downloading dataset from {DATASET_URL}...")
    try:
        urllib.request.urlretrieve(DATASET_URL, RAW_FILE_PATH)
        print(f"Successfully downloaded to {RAW_FILE_PATH}")
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

def preprocess_and_split():
    print("Preprocessing and splitting dataset...")
    # Load dataset, handling '?' as missing values
    df = pd.read_csv(RAW_FILE_PATH, names=COLUMNS, na_values="?")
    
    # Check shape
    print(f"Original dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # Convert target to binary: 0 = no heart disease, 1 = heart disease (values 1, 2, 3, 4)
    # The original dataset contains target values 0 (absence) and 1,2,3,4 (presence)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # Perform train-test split (80-20 stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )

    # Save to processed directory
    train_df.to_csv(TRAIN_FILE_PATH, index=False)
    test_df.to_csv(TEST_FILE_PATH, index=False)
    print(f"Saved train set to {TRAIN_FILE_PATH} ({train_df.shape[0]} rows)")
    print(f"Saved test set to {TEST_FILE_PATH} ({test_df.shape[0]} rows)")

if __name__ == "__main__":
    download_data()
    preprocess_and_split()
