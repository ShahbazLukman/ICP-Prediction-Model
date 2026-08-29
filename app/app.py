
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from predict import predict_lead, load_model, FEATURES


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Course Lead Intelligence",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🎯 Course Lead Intelligence System")

st.caption(
    "Predict which course leads are most likely to purchase "
    "within the next 30 days."
)


# ============================================================
# MODEL CHECK
# ============================================================

MODEL_PATH = ROOT / "models" / "purchase_model.joblib"

if not MODEL_PATH.exists():
    st.warning(
        "⚠️ Model not found.\n\n"
        "Please train the model first using:\n\n"
        "`python src/train.py`"
    )
    st.stop()


# ============================================================
# SIDEBAR - SINGLE LEAD PREDICTION
# ============================================================

st.sidebar.header("📌 Lead Information")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=24
)

education = st.sidebar.selectbox(
    "Education",
    ["High School", "Diploma", "Bachelor", "Master"]
)

occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "Student",
        "Working Professional",
        "Job Seeker",
        "Self Employed"
    ]
)

city = st.sidebar.selectbox(
    "City",
    [
        "Delhi",
        "Mumbai",
        "Bengaluru",
        "Hyderabad",
        "Pune",
        "Chennai",
        "Kolkata",
        "Lucknow",
        "Jaipur",
        "Other"
    ]
)

course = st.sidebar.selectbox(
    "Course Enquiry",
    [
        "Data Science",
        "Finance",
        "Accounting",
        "Digital Marketing",
        "Business Analytics"
    ]
)

source = st.sidebar.selectbox(
    "Lead Source",
    [
        "Google",
        "Instagram",
        "Facebook",
        "Referral",
        "Organic",
        "LinkedIn"
    ]
)


st.sidebar.subheader("📊 Engagement Behaviour")

website_visits = st.sidebar.number_input(
    "Website Visits",
    min_value=0,
    max_value=100,
    value=8
)

course_page_views = st.sidebar.number_input(
    "Course Page Views",
    min_value=0,
    max_value=100,
    value=5
)

pricing_page_views = st.sidebar.number_input(
    "Pricing Page Views",
    min_value=0,
    max_value=50,
    value=2
)

brochure_downloaded = st.sidebar.selectbox(
    "Brochure Downloaded?",
    ["No", "Yes"]
)

email_opens = st.sidebar.number_input(
    "Email Opens",
    min_value=0,
    max_value=100,
    value=4
)

email_clicks = st.sidebar.number_input(
    "Email Clicks",
    min_value=0,
    max_value=50,
    value=2
)

whatsapp_responses = st.sidebar.number_input(
    "WhatsApp Responses",
    min_value=0,
    max_value=50,
    value=2
)

calls_answered = st.sidebar.number_input(
    "Calls Answered",
    min_value=0,
    max_value=20,
    value=1
)

followup_count = st.sidebar.number_input(
    "Follow-up Count",
    min_value=0,
    max_value=30,
    value=2
)

webinar_attended = st.sidebar.selectbox(
    "Webinar Attended?",
    ["No", "Yes"]
)

demo_attended = st.sidebar.selectbox(
    "Demo Attended?",
    ["No", "Yes"]
)

video_watches = st.sidebar.number_input(
    "Video Watches",
    min_value=0,
    max_value=50,
    value=3
)

days_since_last_activity = st.sidebar.number_input(
    "Days Since Last Activity",
    min_value=0.0,
    max_value=180.0,
    value=3.0
)


# Convert Yes/No values to 1/0 for the ML model

brochure_value = 1 if brochure_downloaded == "Yes" else 0
webinar_value = 1 if webinar_attended == "Yes" else 0
demo_value = 1 if demo_attended == "Yes" else 0


# ============================================================
# CREATE LEAD DICTIONARY
# ============================================================

lead = {
    "age": age,
    "education": education,
    "occupation": occupation,
    "city": city,
    "course_enquiry": course,
    "lead_source": source,
    "website_visits": website_visits,
    "course_page_views": course_page_views,
    "pricing_page_views": pricing_page_views,
    "brochure_downloaded": brochure_value,
    "email_opens": email_opens,
    "email_clicks": email_clicks,
    "whatsapp_responses": whatsapp_responses,
    "calls_answered": calls_answered,
    "followup_count": followup_count,
    "webinar_attended": webinar_value,
    "demo_attended": demo_value,
    "video_watches": video_watches,
    "days_since_last_activity": days_since_last_activity
}


# ============================================================
# SINGLE LEAD PREDICTION
# ============================================================

st.subheader("🔮 Single Lead Prediction")

st.write(
    "Enter the information in the sidebar and click the button "
    "below to estimate the probability that this lead will "
    "purchase a course within 30 days."
)

if st.button(
    "🚀 Predict Purchase Probability",
    type="primary",
    use_container_width=True
):

    result = predict_lead(lead)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Purchase Probability",
        f'{result["purchase_probability"]:.1f}%'
    )

    c2.metric(
        "Lead Score",
        f'{result["lead_score"]:.1f}/100'
    )

    c3.metric(
        "Lead Category",
        result["lead_category"]
    )

    st.progress(
        min(result["purchase_probability"] / 100, 1.0)
    )

    if result["lead_category"] == "Hot":

        st.success(
            "🔥 High-priority lead: consider prioritizing "
            "sales follow-up."
        )

    elif result["lead_category"] == "Warm":

        st.info(
            "🟡 Moderate/high purchase intent: consider "
            "targeted follow-up."
        )

    elif result["lead_category"] == "Medium":

        st.warning(
            "🟠 Medium purchase intent: consider nurturing "
            "this lead before intensive sales effort."
        )

    else:

        st.warning(
            "🧊 Lower predicted purchase intent: consider "
            "lower-cost nurturing."
        )


# ============================================================
# DATA FORMAT GUIDE
# ============================================================

st.divider()

with st.expander("ℹ️ How should I prepare my data?"):

    st.markdown(
        """
        ### 📋 Batch Prediction Data Format

        To predict multiple leads at once, your CSV file should
        contain the same feature columns used by the model.

        **Do not include the target column**
        `purchased_within_30_days`.

        The model is trying to predict that value.
        """
    )

    guide = pd.DataFrame({
        "Column": FEATURES,
        "What to enter": [
            "Age of the lead",
            "Education level",
            "Occupation",
            "City",
            "Course the lead enquired about",
            "Where the lead came from",
            "Number of website visits",
            "Number of course page views",
            "Number of pricing page views",
            "1 if brochure downloaded, otherwise 0",
            "Number of emails opened",
            "Number of email links clicked",
            "Number of WhatsApp responses",
            "Number of calls answered",
            "Number of follow-ups",
            "1 if webinar attended, otherwise 0",
            "1 if demo attended, otherwise 0",
            "Number of course videos watched",
            "Days since last activity"
        ]
    })

    st.dataframe(
        guide,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SAMPLE CSV TEMPLATE
# ============================================================

st.subheader("📄 Batch Prediction")

st.write(
    "Want to predict many leads at once? Download the sample "
    "CSV template, fill it with your lead data, and upload it."
)


# Example template

sample_data = pd.DataFrame([
    {
        "age": 23,
        "education": "Bachelor",
        "occupation": "Student",
        "city": "Delhi",
        "course_enquiry": "Data Science",
        "lead_source": "Google",
        "website_visits": 12,
        "course_page_views": 8,
        "pricing_page_views": 4,
        "brochure_downloaded": 1,
        "email_opens": 6,
        "email_clicks": 3,
        "whatsapp_responses": 3,
        "calls_answered": 2,
        "followup_count": 3,
        "webinar_attended": 1,
        "demo_attended": 1,
        "video_watches": 5,
        "days_since_last_activity": 1
    },
    {
        "age": 28,
        "education": "Master",
        "occupation": "Working Professional",
        "city": "Mumbai",
        "course_enquiry": "Finance",
        "lead_source": "LinkedIn",
        "website_visits": 6,
        "course_page_views": 4,
        "pricing_page_views": 2,
        "brochure_downloaded": 1,
        "email_opens": 4,
        "email_clicks": 2,
        "whatsapp_responses": 1,
        "calls_answered": 1,
        "followup_count": 2,
        "webinar_attended": 0,
        "demo_attended": 1,
        "video_watches": 2,
        "days_since_last_activity": 5
    }
])

sample_csv = sample_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Sample CSV Template",
    data=sample_csv,
    file_name="course_lead_prediction_template.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# UPLOAD CSV
# ============================================================

st.markdown("### Step 1 — Download the template")

st.write(
    "Download the template above and open it in Excel, "
    "Google Sheets, or another spreadsheet application."
)

st.markdown("### Step 2 — Add your lead information")

st.write(
    "Replace the example values with your own lead information. "
    "Keep the column names unchanged."
)

st.markdown("### Step 3 — Upload your completed CSV")

uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="Upload a CSV containing the feature columns shown in the data guide."
)


# ============================================================
# BATCH PREDICTION
# ============================================================

if uploaded is not None:

    try:

        batch = pd.read_csv(uploaded)

        st.success(
            f"✅ File uploaded successfully: "
            f"{len(batch):,} leads found."
        )

        # Check required columns

        missing = [
            column
            for column in FEATURES
            if column not in batch.columns
        ]

        if missing:

            st.error(
                "❌ Your CSV is missing the following required "
                "columns:\n\n"
                + "\n".join(f"- {column}" for column in missing)
            )

            st.info(
                "Please download the sample template above and "
                "keep the column names unchanged."
            )

        else:

            # Check for extra columns

            extra_columns = [
                column
                for column in batch.columns
                if column not in FEATURES
            ]

            if extra_columns:

                st.info(
                    "ℹ️ The following extra columns will be ignored "
                    "during prediction: "
                    + ", ".join(extra_columns)
                )

            # Check model

            model = load_model()

            # Predict probabilities

            probabilities = model.predict_proba(
                batch[FEATURES]
            )[:, 1]

            batch["purchase_probability"] = (
                probabilities * 100
            ).round(1)

            batch["lead_score"] = (
                probabilities * 100
            ).round(1)

            # Lead categories

            batch["lead_category"] = pd.cut(
                batch["lead_score"],
                bins=[
                    -1,
                    39.999,
                    59.999,
                    79.999,
                    100
                ],
                labels=[
                    "Cold",
                    "Medium",
                    "Warm",
                    "Hot"
                ]
            )

            # Sort highest probability first

            batch = batch.sort_values(
                "purchase_probability",
                ascending=False
            )

            # ====================================================
            # SUMMARY
            # ====================================================

            st.divider()

            st.subheader("📊 Prediction Summary")

            total_leads = len(batch)

            hot_count = (
                batch["lead_category"] == "Hot"
            ).sum()

            warm_count = (
                batch["lead_category"] == "Warm"
            ).sum()

            medium_count = (
                batch["lead_category"] == "Medium"
            ).sum()

            cold_count = (
                batch["lead_category"] == "Cold"
            ).sum()

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Total Leads",
                f"{total_leads:,}"
            )

            c2.metric(
                "🔥 Hot",
                f"{hot_count:,}"
            )

            c3.metric(
                "🟡 Warm",
                f"{warm_count:,}"
            )

            c4.metric(
                "🟠 Medium",
                f"{medium_count:,}"
            )

            c5.metric(
                "🧊 Cold",
                f"{cold_count:,}"
            )

            # ====================================================
            # RESULTS TABLE
            # ====================================================

            st.subheader("🎯 Scored Leads")

            st.dataframe(
                batch,
                use_container_width=True,
                hide_index=True
            )

            # ====================================================
            # DOWNLOAD RESULTS
            # ====================================================

            result_csv = batch.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Scored Leads",
                data=result_csv,
                file_name="scored_leads.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:

        st.error(
            "❌ Something went wrong while processing the CSV."
        )

        st.code(str(e))


# ============================================================
# FOOTER / MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("🤖 About this project"):

    st.markdown(

        """
        ### Course Lead Purchase Prediction
        This machine-learning application predicts the probability
        that a course lead will purchase a course within 30 days.

        **Prediction target**

        `purchased_within_30_days`

        **Important**

        This model predicts **purchase likelihood** based on
        observed lead behaviour. It does not determine which
        course is best for a student.

        **Example use case**

        A sales team has 10,000 leads but only enough capacity
        to contact 1,000 of them immediately.

        The model can rank leads according to predicted purchase
        probability so the sales team can prioritize follow-ups.

        This project uses a synthetic educational-lead dataset
        for demonstration and learning purposes.
        """
    )