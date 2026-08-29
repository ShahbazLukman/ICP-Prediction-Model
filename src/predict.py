from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "purchase_model.joblib"

FEATURES = [
    "age", "education", "occupation", "city", "course_enquiry", "lead_source",
    "website_visits", "course_page_views", "pricing_page_views",
    "brochure_downloaded", "email_opens", "email_clicks",
    "whatsapp_responses", "calls_answered", "followup_count",
    "webinar_attended", "demo_attended", "video_watches",
    "days_since_last_activity"
]

def load_model():
    return joblib.load(MODEL_PATH)

def predict_lead(lead: dict):
    model = load_model()
    row = pd.DataFrame([lead])[FEATURES]
    probability = float(model.predict_proba(row)[0, 1])
    score = round(probability * 100, 1)
    if score >= 80:
        category = "Hot"
    elif score >= 60:
        category = "Warm"
    elif score >= 40:
        category = "Medium"
    else:
        category = "Cold"
    return {
        "purchase_probability": score,
        "lead_score": score,
        "lead_category": category
    }
