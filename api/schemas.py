from pydantic import BaseModel, Field, ConfigDict

class PredictRequest(BaseModel):
    age: float = Field(..., description="Age in years", json_schema_extra={"example": 52.0})
    sex: float = Field(..., description="Sex (1.0 = male; 0.0 = female)", json_schema_extra={"example": 1.0})
    cp: float = Field(..., description="Chest pain type (1.0, 2.0, 3.0, 4.0)", json_schema_extra={"example": 3.0})
    trestbps: float = Field(..., description="Resting blood pressure (in mm Hg)", json_schema_extra={"example": 125.0})
    chol: float = Field(..., description="Serum cholesterol in mg/dl", json_schema_extra={"example": 212.0})
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1.0 = true; 0.0 = false)", json_schema_extra={"example": 0.0})
    restecg: float = Field(..., description="Resting electrocardiographic results (0.0, 1.0, 2.0)", json_schema_extra={"example": 1.0})
    thalach: float = Field(..., description="Maximum heart rate achieved", json_schema_extra={"example": 168.0})
    exang: float = Field(..., description="Exercise induced angina (1.0 = yes; 0.0 = no)", json_schema_extra={"example": 0.0})
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest", json_schema_extra={"example": 1.0})
    slope: float = Field(..., description="The slope of the peak exercise ST segment (1.0, 2.0, 3.0)", json_schema_extra={"example": 1.0})
    ca: float = Field(..., description="Number of major vessels (0.0-3.0) colored by fluoroscopy", json_schema_extra={"example": 2.0})
    thal: float = Field(..., description="Thalassemia (3.0 = normal; 6.0 = fixed defect; 7.0 = reversible defect)", json_schema_extra={"example": 3.0})

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 52.0,
                "sex": 1.0,
                "cp": 3.0,
                "trestbps": 125.0,
                "chol": 212.0,
                "fbs": 0.0,
                "restecg": 1.0,
                "thalach": 168.0,
                "exang": 0.0,
                "oldpeak": 1.0,
                "slope": 1.0,
                "ca": 2.0,
                "thal": 3.0
            }
        }
    )

class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Heart disease risk prediction (0 = low risk/absence, 1 = high risk/presence)", json_schema_extra={"example": 1})
    confidence: float = Field(..., description="Confidence/probability score of the prediction", json_schema_extra={"example": 0.85})
    model_name: str = Field(..., description="Name of the model used for inference", json_schema_extra={"example": "Random Forest"})
