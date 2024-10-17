from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="House Price Prediction API")

# Load the model
model = joblib.load("regression.joblib")

class HousePredictionRequest(BaseModel):
    size: float
    nb_rooms: int
    garden: int

class HousePredictionResponse(BaseModel):
    predicted_price: float

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API"}

@app.post("/predict", response_model=HousePredictionResponse)
def predict(request: HousePredictionRequest):
    try:
        # Create input array for model
        features = [[
            request.size,
            request.nb_rooms,
            request.garden
        ]]
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return HousePredictionResponse(predicted_price=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)