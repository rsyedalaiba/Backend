from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict_yield

app = FastAPI(title="Crop Yield Prediction API")

# Input schema
class CropInput(BaseModel):
    province: str
    district: str
    crop_type: str
    soil_type: str
    sowing_date: str
    harvest_date: str
    area: float
    year: int
    temperature: float
    rainfall: float
    nitrogen: float
    phosphorus: float
    potassium: float
    soil_ph: float
    ndvi: float

@app.get("/")
def root():
    return {"message": "Crop Yield API is running!"}

@app.post("/predict")
def predict(data: CropInput):
    data_dict = data.dict()
    result = predict_yield(data_dict)
    return result
