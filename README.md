# Heart Disease Risk Classifier: End-to-End MLOps Solution

This repository contains an end-to-end Machine Learning Operations (MLOps) solution for predicting heart disease risk using the Heart Disease UCI Cleveland dataset. The project demonstrates best practices in data acquisition, reproducible preprocessing pipelines, experiment tracking, containerization, serving APIs, automated testing, continuous integration, local Kubernetes deployment, and system monitoring.

---

## 1. Project Architecture

The overall system architecture and workflow are structured as follows:

```
+------------------+     +------------------------+     +--------------------------+
|  UCI Repository  | --> | data/download_dataset  | --> | data/processed (Splits)  |
+------------------+     +------------------------+     +--------------------------+
                                                                     |
                                                                     v
+------------------+     +------------------------+     +--------------------------+
|    MLflow UI     | <-- |      src/train.py      | <-- |   src/preprocessing.py   |
| (Local Tracking) |     |  (GridSearch Tuning)   |     |   (ColumnTransformer)    |
+------------------+     +------------------------+     +--------------------------+
                                     |
                                     v
                          +------------------------+     +--------------------------+
                          |   models/best_model    | --> |       api/main.py        |
                          |      (Serialized)      |     |     (FastAPI Server)     |
                          +------------------------+     +--------------------------+
                                                                     |
                                                                     v
                                                        +--------------------------+
                                                        |  Docker / Kubernetes     |
                                                        |  (Prometheus Metrics)    |
                                                        +--------------------------+
```

---

## 2. Directory Structure

```
mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions CI/CD configuration
├── data/
│   ├── download_dataset.py       # Script to fetch, clean and split the UCI data
│   ├── raw/                      # Raw UCI heart disease csv
│   └── processed/                # Preprocessed train/test csv splits
├── notebooks/
│   ├── eda.py                    # Script used to generate EDA plots
│   └── eda_and_modeling.ipynb    # Jupyter Notebook explaining EDA & initial experiments
├── reports/
│   └── figures/                  # Visualization plots for EDA reporting
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Data scaling, encoding, and imputation pipeline
│   └── train.py                  # Model training and MLflow tracking script
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI serving app with endpoints (/predict, /health, /metrics)
│   └── schemas.py                # Pydantic input and output validation models
├── tests/
│   ├── __init__.py
│   ├── test_data.py              # Pytest file for data ingestion and preprocessing checks
│   └── test_model.py             # Pytest file for API routes and response shapes
├── kubernetes/
│   ├── deployment.yaml           # Deployment configuration (2 replicas, health checks)
│   └── service.yaml              # LoadBalancer service mapping port 80 to 8000
├── Dockerfile                    # Containerization instructions
├── requirements.txt              # Project package requirements
└── README.md                     # This documentation
```

---

## 3. Setup and Installation

### 3.1 Prerequisites
- Python 3.10+
- Pip
- Docker (optional, for containerization)
- Kubectl & Minikube (optional, for orchestration)

### 3.2 Installation Steps
1. Clone the repository to your local workspace.
2. Initialize and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Workflows and Execution

### 4.1 Data Acquisition & Preprocessing
To fetch the Heart Disease Cleveland dataset from the UCI Machine Learning Repository, clean it, split it into 80-20 stratified splits, and save them, run:
```bash
python data/download_dataset.py
```
This downloads `data/raw/heart_disease_raw.csv` and generates `train.csv` and `test.csv` in `data/processed/`.

### 4.2 Exploratory Data Analysis (EDA)
To run the Jupyter Notebook demonstrating EDA and baseline models:
```bash
jupyter notebook notebooks/eda_and_modeling.ipynb
```
The pre-generated visualizations are saved in the `reports/figures/` folder.

### 4.3 Model Training and MLflow Experiment Tracking
To train Logistic Regression and Random Forest models, tune hyperparameters using grid search cross-validation, log experiments to MLflow, and serialize the best pipeline:
```bash
python src/train.py
```
This logs the runs inside the local MLflow server. To open the MLflow dashboard UI and inspect logged parameters, metrics, plots, and models:
```bash
mlflow ui
```
Navigate to `http://localhost:5000` in your web browser. The overall best model is saved at `models/best_model.joblib`.

### 4.4 Running Serving API locally
To serve the serialized model using FastAPI:
```bash
uvicorn api.main:app --reload --port 8000
```
- **Interactive Documentation:** Go to `http://localhost:8000/docs` (Swagger UI) to test endpoints manually.
- **Predict Risk:** Send a POST request to `/predict` using curl:
  ```bash
  curl -X POST "http://localhost:8000/predict" \
       -H "Content-Type: application/json" \
       -d "{\"age\": 52.0, \"sex\": 1.0, \"cp\": 3.0, \"trestbps\": 125.0, \"chol\": 212.0, \"fbs\": 0.0, \"restecg\": 1.0, \"thalach\": 168.0, \"exang\": 0.0, \"oldpeak\": 1.0, \"slope\": 1.0, \"ca\": 2.0, \"thal\": 3.0}"
  ```
- **Health Check:** `http://localhost:8000/health`
- **Prometheus Metrics:** `http://localhost:8000/metrics`

### 4.5 Running Automated Tests
To run unit and integration tests (for preprocessing pipeline, data format, FastAPI server routes, and responses):
```bash
python -m pytest tests/
```

---

## 5. Deployment and Containerization

### 5.1 Docker Containerization
To package the API and model together inside a container:
1. Build the Docker image:
   ```bash
   docker build -t heart-disease-api:latest .
   ```
2. Run the Docker container locally:
   ```bash
   docker run -d -p 8000:8000 --name heart-disease-api-instance heart-disease-api:latest
   ```
The server will now be reachable at `http://localhost:8000/predict`.

### 5.2 Kubernetes Local Deployment
1. Start Minikube (ensure your virtualization daemon is running):
   ```bash
   minikube start
   ```
2. Point your terminal's Docker daemon to Minikube's Docker registry to build the image directly inside the cluster:
   ```bash
   minikube docker-env | Invoke-Expression
   docker build -t heart-disease-api:latest .
   ```
3. Deploy the manifests:
   ```bash
   kubectl apply -f kubernetes/
   ```
4. Verify pods and service status:
   ```bash
   kubectl get pods
   kubectl get service heart-disease-service
   ```
5. Forward the service port to query it locally:
   ```bash
   kubectl port-forward service/heart-disease-service 8080:80
   ```
The API is now served at `http://localhost:8080/predict`.

---

## 6. System Monitoring

The FastAPI server exposes real-world Prometheus system metrics on `http://localhost:8000/metrics`.
- **API Request Counts:** `api_requests_total` partitioned by `endpoint`, `method`, and `http_status`.
- **Request Latency:** `api_request_duration_seconds` histogram metrics tracking response times.
- **Model Output Distributions:** `api_predictions_total` tracking risk outputs (`0` vs `1`).
- **Prediction Confidence:** `api_prediction_confidence` histogram monitoring model probability scores.

To scrape these metrics:
1. Install Prometheus and add the API endpoint under `scrape_configs` in your `prometheus.yml`:
   ```yaml
   scrape_configs:
     - job_name: 'heart-disease-api'
       scrape_interval: 5s
       static_configs:
         - targets: ['localhost:8000'] # Or the Kubernetes service cluster IP
   ```
2. Start Prometheus and Grafana.
3. Import a Grafana dashboard and point it to the Prometheus datasource to monitor request volume, error rates, model outputs, confidence intervals, and response latency.
