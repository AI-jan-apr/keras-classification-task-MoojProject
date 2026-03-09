from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="🏥 Cancer Diagnosis API", 
              description="Breast Cancer Classification using Keras")

 
# Check which model file exists
model_path = None
if os.path.exists('save_models/cancer_model.h5'):
    model = load_model('save_models/cancer_model.h5')
elif os.path.exists('save_models/model_weights.pkl'):
    model = load_model('save_models/model_weights.pkl')

else:

    raise FileNotFoundError("No model file found")


try:
    scaler = joblib.load('save_models/scaler_weights.pkl')

except:

    raise FileNotFoundError("Scaler file not found")

try:
    feature_names = joblib.load('save_models/feature_names.pkl')

except:

    raise FileNotFoundError("Feature names file not found")



class CancerData(BaseModel):

    # First 10 features (mean values)
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    
    # Standard error features
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    
    # Worst features
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


# ============================================
# 3️⃣ API Endpoints
# ============================================

@app.get("/")
def home():
     return {
        "message": "🏥 Cancer Diagnosis API",
 
    }

 

@app.post("/predict")
def predict(data: CancerData):
 
    try:
        # Convert input data to dictionary
        data_dict = data.dict()
        
        # Extract features in correct order

        features = []
        for feature in feature_names:
            if feature in data_dict:
                features.append(data_dict[feature])
            else:
                features.append(0)  # Default if missing
        
        # Verify we have correct number of features
        if len(features) != len(feature_names):
            return {
                "error": f"Expected {len(feature_names)} features, got {len(features)}",
                "status": "❌ Failed"
            }
        
        # Scale features using the loaded scaler
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)[0][0]
        
        # Convert to diagnosis
        diagnosis = "(Malignant)" if prediction > 0.5 else " (Benign)"
        probability = max(prediction, 1 - prediction) * 100
        
        return {
            "diagnosis": diagnosis,
            "probability": f"{probability:.2f}%",
            "score": float(prediction),
            "status": "Success"
        }
    
    except Exception as e:
        return {
            "error": str(e),
        }


import uvicorn
