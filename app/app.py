"""
app.py — Streamlit web app for academic stress prediction.
"""

import os
import sys
import csv
from datetime import datetime

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    predict,
    generate_suggestions,
    ORDINAL_MAPPINGS,
    NOMINAL_CATEGORIES,
)
import style as ui

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Academic Stress Prediction",
    page_icon="🎓",
    layout="wide",
)

ui.inject_css()

# -----------------------------------------------------------------
# Warning banner + hero
# -----------------------------------------------------------------
st.warning(
    "**Research prototype — not for actual stress assessment.** "
    "This model was trained on only 39 survey responses and does not "
    "generalize reliably (test accuracy 0.25). Predictions are "
    "demonstration-only."
)

ui.hero(
    title="Academic Stress Prediction",
    subtitle=(
        "A machine learning study of undergraduate students in Nepal · "
        "BCA final-year research project"
    ),
    eyebrow="Research Prototype",
)

# -----------------------------------------------------------------
# Input form
# -----------------------------------------------------------------
ui.section("Your Information")

with st.form("prediction_form"):
    st.markdown("##### Demographics")
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

    st.markdown("##### Academics")
    col7, col8, col9 = st.columns(3)
    with col7:
        study_hours = st.selectbox(
            "Study hours per day",
            list(ORDINAL_MAPPINGS["study_hours"].keys()),
        )
    with col8:
        attendance = st.selectbox(
            "Class attendance",
            list(ORDINAL_MAPPINGS["attendance"].keys()),
        )
    with col9:
        exam_count = st.selectbox(
            "Major exams this semester",
            list(ORDINAL_MAPPINGS["exam_count"].keys()),
        )

    col10, col11 = st.columns(2)
    with col10:
        assignment_stress = st.slider(
            "Overwhelmed by assignments / projects (1=Never, 5=Very Often)",
            1, 5, 3,
        )
    with col11:
        academic_performance = st.slider(
            "Satisfaction with academic performance (1=Very dissatisfied, 5=Very satisfied)",
            1, 5, 3,
        )

    st.markdown("##### Lifestyle")
    col12, col13, col14 = st.columns(3)
    with col12:
        sleep_hours = st.selectbox(
            "Sleep hours per night",
            list(ORDINAL_MAPPINGS["sleep_hours"].keys()),
        )
    with col13:
        social_media = st.selectbox(
            "Social media per day",
            list(ORDINAL_MAPPINGS["social_media"].keys()),
        )
    with col14:
        screen_time = st.selectbox(
            "Total screen time per day",
            list(ORDINAL_MAPPINGS["screen_time"].keys()),
        )

    col15, col16 = st.columns(2)
    with col15:
        physical_activity = st.selectbox(
            "Physical activity",
            list(ORDINAL_MAPPINGS["physical_activity"].keys()),
        )
    with col16:
        part_time_job = st.selectbox(
            "Part-time job",
            NOMINAL_CATEGORIES["part_time_job"],
        )

    st.markdown("##### Psychosocial")
    col17, col18 = st.columns(2)
    with col17:
        financial_stress = st.slider(
            "Financial concerns cause stress (1=Never, 5=Very Often)",
            1, 5, 3,
        )
        social_support = st.slider(
            "Have someone to talk to (1=Never, 5=Very Often)",
            1, 5, 3,
        )
    with col18:
        career_stress = st.slider(
            "Career concerns cause stress (1=Never, 5=Very Often)",
            1, 5, 3,
        )
        academic_life_satisfaction = st.slider(
            "Satisfaction with academic life (1=Very dissatisfied, 5=Very satisfied)",
            1, 5, 3,
        )
        thought_break = st.selectbox(
            "Thought about taking a break from studies",
            list(ORDINAL_MAPPINGS["thought_break"].keys()),
        )

    st.markdown("")
    submitted = st.form_submit_button(
        "Predict My Stress Level →",
        use_container_width=True,
    )

# -----------------------------------------------------------------
# Modal: results + suggestions
# -----------------------------------------------------------------
@st.dialog("Your Result", width="medium")
def show_result_modal(user_input: dict, label: str, proba: dict):
    # --- Prediction badge ---
    badge_html = ui.prediction_badge(label)
    ui.card(
        f"""
        <h4>Estimated stress level</h4>
        <div style="font-size:1.4rem; margin:0.4rem 0;">{badge_html}</div>
        <p style="color:#6b6b6b; margin:0;">
            This is a demonstration output. Do not use it to make decisions
            about your mental health.
        </p>
        """
    )

    # --- Where the user sits on the scale ---
    scale_positions = {"Low": 0.15, "Moderate": 0.50, "High": 0.85}
    marker_pos = scale_positions.get(label, 0.50)

    st.markdown("**Where your answers place you**")
    st.markdown(
        f"""
        <div style="margin: 0.6rem 0 0.8rem 0;">
            <div style="position: relative; height: 34px;">
                <div style="
                    position: absolute; top: 12px; left: 0; right: 0; height: 8px;
                    background: linear-gradient(90deg,
                        #c8e6c9 0%,
                        #c8e6c9 30%,
                        #ffe082 35%,
                        #ffe082 65%,
                        #ef9a9a 70%,
                        #ef9a9a 100%);
                    border-radius: 4px;
                "></div>
                <div style="
                    position: absolute; top: 0;
                    left: {marker_pos * 100:.0f}%;
                    transform: translateX(-50%);
                    font-size: 22px; line-height: 1;
                ">▼</div>
            </div>
            <div style="
                display: flex; justify-content: space-between;
                font-size: 0.78rem; color: #6b6b6b;
                margin-top: 4px; font-weight: 600;
                letter-spacing: 0.05em;
            ">
                <span>LOW</span><span>MODERATE</span><span>HIGH</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



    # --- Suggestions ---
    st.markdown("---")
    suggestions = generate_suggestions(user_input, label)

    level_titles = {
        "Low": "Maintain your current habits",
        "Moderate": "A few adjustments may help",
        "High": "Prioritize your well-being today",
    }
    ui.render_priority_banner(
        label=f"Stress level · {label}",
        title=level_titles.get(label, "Recommended actions"),
        description=suggestions["summary"],
    )

    if suggestions["tips"]:
        st.markdown("##### Focus areas — start with #01")
        cols = st.columns(2)
        for i, tip in enumerate(suggestions["tips"]):
            with cols[i % 2]:
                ui.render_tip_card(
                    number=i + 1,
                    icon=tip["icon"],
                    category=tip["category"],
                    body=tip["body"],
                    action=tip["action"],
                )
    else:
        st.success(
            "No specific alerts triggered. Keep maintaining your current habits."
        )

    st.markdown("##### Daily habits that help")
    habit_cols = st.columns(2)
    for i, habit in enumerate(suggestions["always"]):
        with habit_cols[i % 2]:
            ui.render_habit_card(
                icon=habit["icon"],
                title=habit["title"],
                body=habit["body"],
            )

    ui.render_help_card()

    # --- Contribute to research ---
    st.markdown("---")
    if st.button("Contribute this response to research (anonymous)"):
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "collected_data", "responses.csv",
        )
        file_exists = os.path.exists(save_path)

        record = dict(user_input)
        record["predicted_label"] = label
        record["timestamp"] = datetime.utcnow().isoformat()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

        st.success("Response saved anonymously. Thank you.")

# -----------------------------------------------------------------
# On submit → open the modal
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
    show_result_modal(user_input, label, proba)

# -----------------------------------------------------------------
# Footer
# -----------------------------------------------------------------
ui.footer()