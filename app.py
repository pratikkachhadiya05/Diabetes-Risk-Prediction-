from pathlib import Path
import shap
import pickle
import uuid
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_rf_model.pkl"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

prediction_store: dict[str, dict] = {}


class InputData(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: int
    bmi: float
    hba1c_level: float
    blood_glucose_level: float


model = None
rf_model = None
try:
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    print(type(model))
    rf_model = model.best_estimator_
    rf_classifier = rf_model.named_steps["rf"]
    print(rf_model.named_steps)
    print(type(rf_model))

    print("Model loaded successfully")
except FileNotFoundError:
    print(f"Error: model file not found at {MODEL_PATH}")
except Exception as exc:
    print(f"Error loading model: {exc}")


def smoking_history_label(value: int) -> str:
    labels = {0: "never", 1: "former", 2: "current"}
    return labels.get(value, "No Info")


def build_model_input(data: InputData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": [data.gender],
            "age": [data.age],
            "hypertension": [data.hypertension],
            "heart_disease": [data.heart_disease],
            "smoking_history": [smoking_history_label(data.smoking_history)],
            "bmi": [data.bmi],
            "HbA1c_level": [data.hba1c_level],
            "blood_glucose_level": [data.blood_glucose_level],
        }
    )


def normalize_bmi(value: float) -> int:
    if value < 18.5:
        return 30
    if value < 25:
        return 20
    if value < 30:
        return 50
    return 80


def normalize_hba1c(value: float) -> int:
    if value < 5.7:
        return 20
    if value < 6.5:
        return 50
    return 80


def normalize_glucose(value: float) -> int:
    if value < 100:
        return 20
    if value < 126:
        return 50
    return 80


def normalize_lifestyle(smoking_history: int) -> int:
    if smoking_history == 0:
        return 20
    if smoking_history == 1:
        return 45
    return 75


def get_chart_data(data: InputData, prob0: float, prob1: float, risk_pct: float) -> dict:
    return {
        "gauge": {"riskPct": risk_pct, "safeZone": round(100 - risk_pct, 1)},
        "probability_bar": {
            "no_diabetes_pct": round(prob0 * 100, 1),
            "diabetes_pct": round(prob1 * 100, 1),
        },
        "radar": {
            "labels": ["BMI", "HbA1c", "Blood Glucose", "Age Impact", "Lifestyle"],
            "user_values": [
                normalize_bmi(data.bmi),
                normalize_hba1c(data.hba1c_level),
                normalize_glucose(data.blood_glucose_level),
                min(100, round(data.age, 1)),
                normalize_lifestyle(data.smoking_history),
            ],
            "normal_values": [20, 20, 20, 35, 20],
        },
    }


def get_recommendations(prediction: int, bmi: float, hba1c: float, glucose: float) -> list[str]:
    recommendations = []

    if prediction == 1:
        recommendations.append("Schedule an appointment with a healthcare provider.")
        recommendations.append("Get a comprehensive diabetes screening test.")
        recommendations.append("Discuss medication and monitoring options with your doctor.")

        if bmi > 25:
            recommendations.append("Consider a weight management plan with a target BMI below 25.")
        if hba1c > 6.5:
            recommendations.append("Monitor blood sugar levels regularly.")
        if glucose > 126:
            recommendations.append("Follow a diabetes-friendly diet plan.")
    else:
        recommendations.append("Maintain a balanced diet rich in vegetables and whole grains.")
        recommendations.append("Exercise regularly, aiming for at least 150 minutes per week.")

        if bmi > 25:
            recommendations.append("Work toward a healthy BMI range.")
        if hba1c > 5.7:
            recommendations.append("Monitor HbA1c levels annually.")
        if glucose > 100:
            recommendations.append("Reduce added sugar and processed foods.")

        recommendations.append("Schedule regular health checkups every 6 months.")

    return recommendations


def run_prediction(data: InputData) -> dict:
    input_df = build_model_input(data)
    X_transformed = rf_model.named_steps["tf"].transform(input_df)
    prediction = int(rf_model.predict(input_df)[0])
    prediction_prob = rf_model.predict_proba(input_df)[0]

    prob0 = round(float(prediction_prob[0]), 4)
    prob1 = round(float(prediction_prob[1]), 4)
    risk_pct = round(prob1 * 100, 1)
    chart_data = get_chart_data(data, prob0, prob1, risk_pct)

    feature_names = rf_model.named_steps["tf"].get_feature_names_out()

    feature_importance = rf_classifier.feature_importances_

    importance = []

    for feature, score in zip(feature_names, feature_importance):

        feature = feature.replace("ohe__", "")
        feature = feature.replace("scaler__", "")
        feature = feature.replace("remainder__", "")
        feature = feature.replace("_", " ").title()

        importance.append({
            "feature": feature,
            "impact": round(float(score * 100), 2),
            "direction": "Important"
        })

    importance = sorted(
        importance,
        key=lambda x: x["impact"],
        reverse=True
    )

    top_features = importance[:5]

    print(top_features)

    return {
        "prediction": prediction,
        "probability_class_0": f"{prob0 * 100:.2f}%",
        "probability_class_1": f"{prob1 * 100:.2f}%",
        "prob0": prob0,
        "prob1": prob1,
        "risk_percentage": risk_pct,
        "riskPct": risk_pct,
        "message": (
            "Please consult a healthcare provider immediately for proper diagnosis and treatment."
            if prediction == 1
            else "Great news! Keep maintaining healthy habits with regular exercise and a balanced diet."
        ),

        "chart_data": chart_data,
        "recommendations": get_recommendations(
            prediction,
            data.bmi,
            data.hba1c_level,
            data.blood_glucose_level,
        ),

        "gender": data.gender,
        "age": data.age,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "smoking_history": data.smoking_history,
        "bmi": data.bmi,
        "hba1c_level": data.hba1c_level,
        "blood_glucose_level": data.blood_glucose_level,

        "top_features": top_features,

    }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="home.html", context={"request": request})


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html", context={"request": request})


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(request=request, name="about.html", context={"request": request})


@app.get("/services", response_class=HTMLResponse)
async def services(request: Request):
    return templates.TemplateResponse(request=request, name="services.html", context={"request": request})


@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request):
    return templates.TemplateResponse(request=request, name="prediction.html", context={"request": request})


@app.post("/predict")
async def predict(
    request: Request,
    gender: str = Form(...),
    age: int = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    smoking_history: int = Form(...),
    bmi: float = Form(...),
    hba1c_level: float = Form(...),
    blood_glucose_level: float = Form(...),
):
    if model is None:
        return templates.TemplateResponse(
            request=request,
            name="prediction.html",
            context={"request": request, "error": "Model not loaded on the server."},
        )

    data = InputData(
        gender=gender,
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        smoking_history=smoking_history,
        bmi=bmi,
        hba1c_level=hba1c_level,
        blood_glucose_level=blood_glucose_level,
    )

    try:
        result = run_prediction(data)
        prediction_id = str(uuid.uuid4())
        prediction_store[prediction_id] = result
        return RedirectResponse(url=f"/result?prediction_id={prediction_id}", status_code=303)
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="prediction.html",
            context={"request": request, "error": str(exc)},
        )


@app.get("/result", response_class=HTMLResponse)
async def show_results(request: Request, prediction_id: str | None = None):
    if not prediction_id:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "request": request,
                "error": "No prediction data found. Please submit the prediction form first.",
            },
        )

    result = prediction_store.get(prediction_id)
    if not result:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "request": request,
                "error": "Prediction not found. It may have expired — please submit a new prediction.",
            },
        )

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={"request": request, **result},
    )


@app.post("/api/predict")
async def api_predict(data: InputData):
    if model is None:
        return JSONResponse(
            content={"error": "Model not loaded on the server."},
            status_code=500,
        )

    try:
        result = run_prediction(data)
        return {
            "prediction": result["prediction"],
            "prob0": result["prob0"],
            "prob1": result["prob1"],
            "probability_class_0": result["prob0"],
            "probability_class_1": result["prob1"],
            "riskPct": result["riskPct"],
            "chart_data": result["chart_data"],
            "recommendations": result["recommendations"],
        }
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
