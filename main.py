from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = FastAPI(title="Maternal Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# API KEY
# ===============================
API_KEY = "1234sishu"

# ===============================
# LOAD MODELS
# ===============================

model1 = joblib.load("models/model1_vitals.pkl")
model2 = joblib.load("models/model2_vitals_symptoms.pkl")
model3 = joblib.load("models/model3_symptoms.pkl")

imputer1 = joblib.load("models/imputer1.pkl")
imputer2 = joblib.load("models/imputer2.pkl")
imputer3 = joblib.load("models/imputer3.pkl")

le1 = joblib.load("models/encoder1.pkl")
le2 = joblib.load("models/encoder2.pkl")
le3 = joblib.load("models/encoder3.pkl")

features1 = joblib.load("models/features1.pkl")
features2 = joblib.load("models/features2.pkl")
features3 = joblib.load("models/features3.pkl")

# ===============================
# VERIFY API KEY
# ===============================

def verify_api_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


# ===============================
# HOME ROUTE
# ===============================

@app.get("/")
def home():
    return {"message": "Maternal Risk Prediction API is running"}


# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_risk(data):

    # Model 1 — Vitals + History
    input1 = pd.DataFrame([[
        data["age"], data["systolic_bp"], data["diastolic_bp"],
        data["blood_sugar"], data["body_temp"],
        data["bmi"], data["prev_complications"],
        data["preexisting_diabetes"], data["gestational_diabetes"],
        data["mental_health"], data["heart_rate"]
    ]], columns=features1)

    x1 = imputer1.transform(input1)
    prob1 = model1.predict_proba(x1)[0]
    pred1 = le1.inverse_transform([model1.predict(x1)[0]])[0]

    prob1_dict = dict(zip(le1.classes_, prob1))
    m1_high = prob1_dict.get("High", 0)
    m1_medium = prob1_dict.get("Medium", 0)
    m1_low = prob1_dict.get("Low", 0)


    # Model 2 — Vitals + Symptoms
    input2 = pd.DataFrame([[
        data["age"], data["systolic_bp"], data["diastolic_bp"],
        data["body_temp"], data["heart_rate"],
        data["vaginal_bleeding"], data["abdominal_pain"],
        data["fever_symptom"], data["dizziness"],
        data["back_pain"], data["pelvic_cramps"],
        data["weakness"]
    ]], columns=features2)

    x2 = imputer2.transform(input2)
    prob2 = model2.predict_proba(x2)[0]
    pred2 = le2.inverse_transform([model2.predict(x2)[0]])[0]

    prob2_dict = dict(zip(le2.classes_, prob2))
    m2_high = prob2_dict.get("High", 0)
    m2_medium = prob2_dict.get("Medium", 0)
    m2_low = prob2_dict.get("Low", 0)


    # Model 3 — Symptoms Only
    input3 = pd.DataFrame([[
        data["vaginal_bleeding"], data["abdominal_pain"],
        data["fever_symptom"], data["dizziness"],
        data["back_pain"], data["pelvic_cramps"],
        data["weakness"]
    ]], columns=features3)

    x3 = imputer3.transform(input3)
    prob3 = model3.predict_proba(x3)[0]
    pred3 = le3.inverse_transform([model3.predict(x3)[0]])[0]

    prob3_dict = dict(zip(le3.classes_, prob3))
    m3_high = prob3_dict.get("High", 0)
    m3_medium = prob3_dict.get("Medium", 0)
    m3_low = prob3_dict.get("Low", 0)


    # ===============================
    # ENSEMBLE WEIGHTS
    # ===============================

    W1, W2, W3 = 0.50, 0.30, 0.20

    final_high = W1*m1_high + W2*m2_high + W3*m3_high
    final_medium = W1*m1_medium + W2*m2_medium + W3*m3_medium
    final_low = W1*m1_low + W2*m2_low + W3*m3_low

    total = final_high + final_medium + final_low

    if total > 0:
        final_high /= total
        final_medium /= total
        final_low /= total

    probs = {
        "High": final_high,
        "Medium": final_medium,
        "Low": final_low
    }

    final_risk = max(probs, key=probs.get)
    confidence = probs[final_risk]

    return {
        "final_risk_level": final_risk,
        "high_risk_percent": round(final_high*100,1),
        "medium_risk_percent": round(final_medium*100,1),
        "low_risk_percent": round(final_low*100,1),
        "confidence_percent": round(confidence*100,1),

        "model1_prediction": pred1,
        "model2_prediction": pred2,
        "model3_prediction": pred3
    }


# ===============================
# API ENDPOINT
# ===============================

@app.post("/predict")
def predict(data: dict, api_key: str = Header(...)):

    verify_api_key(api_key)

    result = predict_risk(data)

    return result
