"""
app.py — Streamlit web app for academic stress prediction.

This is the main entry point. Users fill out a form with their
demographic, academic, and lifestyle information, and the trained
Decision Tree model predicts their academic stress level.

IMPORTANT: The model was trained on only 39 responses. Predictions
are UNRELIABLE and for demonstration only. A large-warning banner
communicates this to users.
"""

import os
import sys
import csv
from datetime import datetime

import streamlit as st

# Make utils importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import predict, ORDINAL_MAPPINGS, NOMINAL_CATEGORIES

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Academic Stress Prediction — Research Prototype",
    page_icon="🎓",
    layout="wide",
)

# -----------------------------------------------------------------
# Warning banner
# -----------------------------------------------------------------
st.warning(
    "⚠️ **Research prototype — NOT for actual stress assessment.** "
    "This model was trained on only 39 survey responses and performs "
    "poorly on unseen data (test accuracy 0.25). Predictions are "
    "unreliable. If you are experiencing significant academic stress, "
    "please speak to a counselor or trusted person."
)

# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.title("🎓 Academic Stress Prediction")
st.markdown(
    "**Machine Learning-Based Prediction of Academic Stress Among "
    "Undergraduate Students in Nepal**  \n"
    "*BCA Final-Year Research Project*"
)

st.markdown("---")

# -----------------------------------------------------------------
# Input form
# -----------------------------------------------------------------
st.subheader("📋 Your Information")

with st.form("prediction_form"):

    # --- Demographics ---
    st.markdown("### Demographics")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox("Age group", list(ORDINAL_MAPPINGS["age"].keys()))
    with col2:
        gender = st.selectbox("Gender", NOMINAL_CATEGORIES["gender"])
    with col3:
        university = st.selectbox("University", NOMINAL_CATEGORIES["university"])

    col4, col5, col6 = st.columns(3)
    with col4:
        program = st.selectbox("Program", NOMINAL_CATEGORIES["program"])
    with col5:
        year = st.selectbox("Year", list(ORDINAL_MAPPINGS["year"].keys()))
    with col6:
        cgpa = st.selectbox("CGPA range", list(ORDINAL_MAPPINGS["cgpa"].keys()))

    # --- Academics ---
    st.markdown("### Academics")
    col7, col8, col9 = st.columns(3)

    with col7:
        study_hours = st.selectbox("Study hours per day (outside class)",
                                    list(ORDINAL_MAPPINGS["study_hours"].keys()))
    with col8:
        attendance = st.selectbox("Class attendance",
                                   list(ORDINAL_MAPPINGS["attendance"].keys()))
    with col9:
        exam_count = st.selectbox("Major exams this semester",
                                    list(ORDINAL_MAPPINGS["exam_count"].keys()))

    col10, col11 = st.columns(2)
    with col10:
        assignment_stress = st.slider("Feeling overwhelmed by assignments/projects (1=Never, 5=Very Often)", 1, 5, 3)
    with col11:
        academic_performance = st.slider("Satisfaction with current academic performance (1=Very dissatisfied, 5=Very satisfied)", 1, 5, 3)

    # --- Lifestyle ---
    st.markdown("### Lifestyle")
    col12, col13, col14 = st.columns(3)

    with col12:
        sleep_hours = st.selectbox("Sleep hours per night",
                                    list(ORDINAL_MAPPINGS["sleep_hours"].keys()))
    with col13:
        social_media = st.selectbox("Social media time per day",
                                     list(ORDINAL_MAPPINGS["social_media"].keys()))
    with col14:
        screen_time = st.selectbox("Total screen time per day",
                                    list(ORDINAL_MAPPINGS["screen_time"].keys()))

    col15, col16 = st.columns(2)
    with col15:
        physical_activity = st.selectbox("Physical activity",
                                          list(ORDINAL_MAPPINGS["physical_activity"].keys()))
    with col16:
        part_time_job = st.selectbox("Part-time job", NOMINAL_CATEGORIES["part_time_job"])

    # --- Psychosocial ---
    st.markdown("### Psychosocial")
    col17, col18 = st.columns(2)

    with col17:
        financial_stress = st.slider("Financial concerns cause stress (1=Never, 5=Very Often)", 1, 5, 3)
        social_support   = st.slider("Have someone to talk to when stressed (1=Never, 5=Very Often)", 1, 5, 3)
    with col18:
        career_stress = st.slider("Career/employment concerns cause stress (1=Never, 5=Very Often)", 1, 5, 3)
        academic_life_satisfaction = st.slider("Satisfaction with academic life (1=Very dissatisfied, 5=Very satisfied)", 1, 5, 3)
        thought_break = st.selectbox("Thought about taking a break from studies",
                                      list(ORDINAL_MAPPINGS["thought_break"].keys()))

    # --- Submit ---
    st.markdown("---")
    submitted = st.form_submit_button("🔮 Predict My Stress Level", use_container_width=True)

# -----------------------------------------------------------------
# Handle submission
# -----------------------------------------------------------------
if submitted:
    user_input = {
        "age": age, "gender": gender, "university": university,
        "program": program, "year": year, "cgpa": cgpa,
        "study_hours": study_hours, "attendance": attendance,
        "exam_count": exam_count, "assignment_stress": assignment_stress,
        "academic_performance": academic_performance,
        "sleep_hours": sleep_hours, "social_media": social_media,
        "screen_time": screen_time, "physical_activity": physical_activity,
        "part_time_job": part_time_job,
        "financial_stress": financial_stress, "social_support": social_support,
        "career_stress": career_stress,
        "academic_life_satisfaction": academic_life_satisfaction,
        "thought_break": thought_break,
    }

    label, proba = predict(user_input)

    st.markdown("---")
    st.subheader("🔮 Prediction Result")

    # Color-code the result
    color_map = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
    st.markdown(f"## {color_map.get(label, '⚪')} Predicted stress level: **{label}**")

    # Show probabilities
    st.markdown("### Class probabilities")
    for cls in ["Low", "Moderate", "High"]:
        st.progress(proba.get(cls, 0.0), text=f"{cls}: {proba.get(cls, 0.0)*100:.1f}%")

    st.error(
        "⚠️ **This prediction is not reliable.** The model was trained "
        "on only 39 responses. Please do not use this output to make "
        "decisions about your mental health. Consult a counselor if you "
        "are struggling."
    )

    # -------- Optionally: save for research --------
    with st.expander("📥 Contribute this response to research (optional)"):
        st.markdown(
            "Your response can help improve this model. It will be stored "
            "anonymously and used only for academic research. "
            "You can skip this."
        )
        if st.button("Save my response anonymously"):
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "collected_data", "responses.csv")
            file_exists = os.path.exists(save_path)

            # We include only the input features + predicted label.
            # NO personal identifiers are stored.
            record = dict(user_input)
            record["predicted_label"] = label
            record["timestamp"] = datetime.utcnow().isoformat()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)

            st.success("✅ Response saved anonymously. Thank you!")

# -----------------------------------------------------------------
# Footer
# -----------------------------------------------------------------
st.markdown("---")
st.caption(
    "Saroj Lamichhane and Santosh Shrestha · BCA Final-Year Project · Tribhuvan University · 2025–2026  \n"
    "Code: github.com/iamsaroj2058/Academic-stress-prediction-ml"
)