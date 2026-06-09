from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import matplotlib as plt
import pickle

app = FastAPI()

#load the model
try:
    with open('diabetes_rf_model','rb') as f:
        model = pickle.load(f)
        print("✅ succefully loaded model")
except FileNotFoundError:
    print("Error :  Model Not Found")


#  templates directory
templates = Jinja2Templates(directory="templates")


#pydantic model for input data
class InputData(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: int
    bmi: float
    hba1c_level: float
    blood_glucose_level: float




@app.get("/home",response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")

@app.get("/about",response_class=HTMLResponse)
async def about(request:Request):
    return templates.TemplateResponse(request=Request, name="about.html")


@app.get("/prediction",response_class=HTMLResponse)
async def prediction(request:Request):
    return templates.TemplateResponse(request=Request, name="about.html")

@app.post('/prediction')
async def predict(data:InputData):

    if model is None:
        return {"error": "Model not loaded on the server."}
    
    input_df = pd.DataFrame({

    "gender": [data.gender],
    "age": [data.age],
    "hypertension": [data.hypertension],
    "heart_disease": [data.heart_disease],
    "smoking_history": [data.smoking_history],
    "bmi": [data.bmi],
    "hba1c_level": [data.hba1c_level],
    "blood_glucose_level": [data.blood_glucose_level]
    })

    try:
        prediction = model.predict(input_df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}
    
    try:
        predict_prob = model.predict_proba(input_df)[0][1]
        return {"prediction": int(prediction), "probability": float(predict_prob)}  
    except Exception as e:
        return {"error": str(e)}
    

    

    

    return {
            "prediction": int(prediction),
            "probability_class_1": float(f"{predict_prob[1] * 100:.2f}%"),
            "probability_class_0": float(f"{predict_prob[0] * 100:.2f}%") 
            }
     
    