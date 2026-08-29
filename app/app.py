
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from predict import predict_lead

st.set_page_config(page_title="Course Lead Intelligence", page_icon="🎯", layout="wide")

st.title("🎯 Course Lead Intelligence System")
st.caption("Synthetic educational-lead dataset • Predicts 30-day purchase probability")

if not (ROOT / "models" / "purchase_model.joblib").exists():
    st.warning("Model not found. Run: python src/train.py")
    st.stop()

st.sidebar.header("Lead Information")

age = st.sidebar.number_input("Age", min_value=18, max_value=70, value=24)
education = st.sidebar.selectbox("Education", ["High School", "Diploma", "Bachelor", "Master"])
occupation = st.sidebar.selectbox("Occupation", ["Student", "Working Professional", "Job Seeker", "Self Employed"])
city = st.sidebar.selectbox("City", ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Kolkata", "Lucknow", "Jaipur", "Other"])
course = st.sidebar.selectbox("Course Enquiry", ["Data Science", "Finance", "Accounting", "Digital Marketing", "Business Analytics"])
source = st.sidebar.selectbox("Lead Source", ["Google", "Instagram", "Facebook", "Referral", "Organic", "LinkedIn"])

website_visits = st.sidebar.number_input("Website Visits", 0, 100, 8)
course_page_views = st.sidebar.number_input("Course Page Views", 0, 100, 5)
pricing_page_views = st.sidebar.number_input("Pricing Page Views", 0, 50, 2)
brochure_downloaded = st.sidebar.selectbox("Brochure Downloaded", [0, 1])
email_opens = st.sidebar.number_input("Email Opens", 0, 100, 4)
email_clicks = st.sidebar.number_input("Email Clicks", 0, 50, 2)
whatsapp_responses = st.sidebar.number_input("WhatsApp Responses", 0, 50, 2)
calls_answered = st.sidebar.number_input("Calls Answered", 0, 20, 1)
followup_count = st.sidebar.number_input("Follow-up Count", 0, 30, 2)
webinar_attended = st.sidebar.selectbox("Webinar Attended", [0, 1])
demo_attended = st.sidebar.selectbox("Demo Attended", [0, 1])
video_watches = st.sidebar.number_input("Video Watches", 0, 50, 3)
days_since_last_activity = st.sidebar.number_input("Days Since Last Activity", 0.0, 180.0, 3.0)

lead = {
    "age": age, "education": education, "occupation": occupation, "city": city,
    "course_enquiry": course, "lead_source": source,
    "website_visits": website_visits, "course_page_views": course_page_views,
    "pricing_page_views": pricing_page_views, "brochure_downloaded": brochure_downloaded,
    "email_opens": email_opens, "email_clicks": email_clicks,
    "whatsapp_responses": whatsapp_responses, "calls_answered": calls_answered,
    "followup_count": followup_count, "webinar_attended": webinar_attended,
    "demo_attended": demo_attended, "video_watches": video_watches,
    "days_since_last_activity": days_since_last_activity
}

if st.button("Predict Purchase Probability", type="primary", use_container_width=True):
    result = predict_lead(lead)

    c1, c2, c3 = st.columns(3)
    c1.metric("Purchase Probability", f'{result["purchase_probability"]:.1f}%')
    c2.metric("Lead Score", f'{result["lead_score"]:.1f}/100')
    c3.metric("Lead Category", result["lead_category"])

    st.progress(min(result["purchase_probability"] / 100, 1.0))

    if result["lead_category"] == "Hot":
        st.success("🔥 High-priority lead: consider prioritizing sales follow-up.")
    elif result["lead_category"] == "Warm":
        st.info("🟡 Moderate/high intent: consider targeted follow-up.")
    else:
        st.warning("🧊 Lower predicted intent: use lower-cost nurturing.")

st.divider()
st.subheader("Batch Scoring")
uploaded = st.file_uploader("Upload a CSV containing the same feature columns", type=["csv"])

if uploaded:
    from predict import load_model, FEATURES
    model = load_model()
    batch = pd.read_csv(uploaded)

    missing = [c for c in FEATURES if c not in batch.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
    else:
        probabilities = model.predict_proba(batch[FEATURES])[:, 1]
        batch["purchase_probability"] = probabilities
        batch["lead_score"] = (probabilities * 100).round(1)
        batch["lead_category"] = pd.cut(
            batch["lead_score"],
            bins=[-1, 39.999, 59.999, 79.999, 100],
            labels=["Cold", "Medium", "Warm", "Hot"]
        )
        batch = batch.sort_values("lead_score", ascending=False)
        st.dataframe(batch, use_container_width=True)
        csv = batch.to_csv(index=False).encode("utf-8")
        st.download_button("Download Scored Leads", csv, "scored_leads.csv", "text/csv")
