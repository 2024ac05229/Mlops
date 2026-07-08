import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "figure.titlesize": 16})

# Paths
NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(NOTEBOOKS_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "raw", "heart_disease_raw.csv")
PLOTS_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load raw data
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = pd.read_csv(DATA_PATH, names=COLUMNS, na_values="?")

# Clean target for EDA (0 = No Disease, 1 = Disease)
df["target_binary"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
df["target_label"] = df["target_binary"].map({0: "No Disease", 1: "Disease"})

def plot_class_balance():
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(x="target_label", data=df, palette="Set2", ax=ax)
    ax.set_title("Class Balance (Heart Disease Presence)")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Count")
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_balance.png"), dpi=150)
    plt.close()
    print("Saved class balance plot.")

def plot_age_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="age", hue="target_label", multiple="stack", kde=True, palette="coolwarm", ax=ax)
    ax.set_title("Age Distribution by Heart Disease Diagnosis")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "age_distribution.png"), dpi=150)
    plt.close()
    print("Saved age distribution plot.")

def plot_correlation_heatmap():
    # Only compute correlation for numerical features plus binary target
    num_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "target_binary"]
    corr = df[num_features].corr()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("Saved correlation heatmap.")

def plot_feature_relationships():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Max Heart Rate (thalach) vs Target
    sns.boxplot(x="target_label", y="thalach", data=df, palette="Set2", ax=axes[0])
    axes[0].set_title("Max Heart Rate by Diagnosis")
    axes[0].set_xlabel("Diagnosis")
    axes[0].set_ylabel("Max Heart Rate (thalach)")
    
    # ST Depression (oldpeak) vs Target
    sns.boxplot(x="target_label", y="oldpeak", data=df, palette="Set2", ax=axes[1])
    axes[1].set_title("ST Depression (oldpeak) by Diagnosis")
    axes[1].set_xlabel("Diagnosis")
    axes[1].set_ylabel("ST Depression (oldpeak)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_relationships.png"), dpi=150)
    plt.close()
    print("Saved feature relationships plot.")

if __name__ == "__main__":
    plot_class_balance()
    plot_age_distribution()
    plot_correlation_heatmap()
    plot_feature_relationships()
