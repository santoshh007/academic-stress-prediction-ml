"""
utils.py — Shared helpers for the Streamlit app.

Provides:
  - preprocess_input(): raw responses -> 37-feature DataFrame
  - predict():          raw responses -> (label, probabilities)
  - generate_suggestions(): personalized stress-reduction tips
"""

import os
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
APP_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(APP_DIR)

MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
SCALER_PATH   = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "X_features.csv")

# ---------------------------------------------------------------
# Load model + scaler
# ---------------------------------------------------------------
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_COLUMNS = pd.read_csv(FEATURES_PATH, nrows=0).columns.tolist()

# ---------------------------------------------------------------
# Ordinal mappings
# ---------------------------------------------------------------
ORDINAL_MAPPINGS = {
    "age": {
        "17-19": 1, "20-22": 2, "23-25": 3, "Above 25": 4,
    },
    "year": {
        "1st Year": 1, "2nd Year": 2, "3rd Year": 3,
        "4th Year": 4, "5th Year": 5,
    },
    "cgpa": {
        "Not Available Yet": 0, "2.0–2.49": 1, "2.5–2.99": 2,
        "3.0–3.49": 3, "3.5–4.0": 4,
    },
    "study_hours": {
        "Less than 1": 1, "1–2": 2, "3–4": 3,
        "5–6": 4, "More than 6 hours": 5,
    },
    "attendance": {
        "Below 60%": 1, "60–70%": 2, "71–80%": 3,
        "81–90%": 4, "Above 90%": 5,
    },
    "exam_count": {
        "1-2": 1, "3-4": 2, "5-6": 3, "More than 6": 4,
    },
    "sleep_hours": {
        "Less than 5": 1, "5–6": 2, "7–8": 3, "More than 8": 4,
    },
    "social_media": {
        "Less than 1 hour": 1, "1–2 hours": 2, "3–4 hours": 3,
        "5–6 hours": 4, "More than 6 hours": 5,
    },
    "physical_activity": {
        "Never": 0, "1–2 days/week": 1, "3–4 days/week": 2,
        "5–6 days/week": 3, "Daily": 4,
    },
    "screen_time": {
        "Less than 3 hours": 1, "3–5 hours": 2,
        "6–8 hours": 3, "More than 8 hours": 4,
    },
    "thought_break": {
        "Never": 1, "Rarely": 2, "Sometimes": 3,
        "Often": 4, "Very Often": 5,
    },
}

NOMINAL_CATEGORIES = {
    "gender": ["Female", "Male", "Prefer not to say"],
    "university": [
        "Kathmandu University", "Other", "Pokhara University",
        "Purbanchal University", "Rajarshi Janak University",
        "Sudurpaschim University", "Tribhuvan University",
    ],
    "program": [
        "BBS", "BCA", "BDS", "BE/B.Tech", "BHM", "BIM",
        "BSc CSIT", "BSc Nursing", "Law(LLB)", "MBBS",
    ],
    "part_time_job": ["No", "Yes"],
}


def _one_hot(base_name, value, categories, drop_first=True):
    cats = categories[1:] if drop_first else categories
    return {f"{base_name}_{c}": (1 if value == c else 0) for c in cats}


def preprocess_input(user_input: dict) -> pd.DataFrame:
    row = {}

    for col, mapping in ORDINAL_MAPPINGS.items():
        row[col] = mapping[user_input[col]]

    for col in [
        "assignment_stress", "financial_stress", "social_support",
        "academic_life_satisfaction", "career_stress", "academic_performance",
    ]:
        row[col] = int(user_input[col])

    row.update(_one_hot("gender", user_input["gender"], NOMINAL_CATEGORIES["gender"]))
    row.update(_one_hot("university", user_input["university"], NOMINAL_CATEGORIES["university"]))
    row.update(_one_hot("program", user_input["program"], NOMINAL_CATEGORIES["program"]))
    row.update(_one_hot("part_time_job", user_input["part_time_job"], NOMINAL_CATEGORIES["part_time_job"]))

    df = pd.DataFrame([row])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


def predict(user_input: dict):
    X = preprocess_input(user_input)
    X_scaled = scaler.transform(X)
    label = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    proba_dict = {cls: float(p) for cls, p in zip(model.classes_, proba)}
    return label, proba_dict


# ---------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------
LEVEL_MESSAGES = {
    "Low": {
        "headline": "🟢 Your responses suggest LOW academic stress.",
        "summary": (
            "Your current habits and academic life appear balanced. "
            "Keep doing what works for you."
        ),
    },
    "Moderate": {
        "headline": "🟡 Your responses suggest MODERATE academic stress.",
        "summary": (
            "You are showing some signs of academic stress. This is common "
            "among undergraduate students, and small adjustments to daily "
            "habits often help."
        ),
    },
    "High": {
        "headline": "🔴 Your responses suggest HIGH academic stress.",
        "summary": (
            "Your answers indicate elevated academic stress. Please take "
            "this seriously. Talking to a counselor, mentor, or trusted "
            "person is strongly recommended."
        ),
    },
}


FEATURE_TIPS = [
    {
        "key": "sleep_hours",
        "condition": lambda v: v == "Less than 5",
        "icon": "🌙",
        "category": "Sleep",
        "body": "You reported sleeping less than 5 hours per night. "
                "Sleep loss is one of the strongest drivers of academic stress.",
        "action": "Aim for 7–8 hours of sleep; keep a consistent bedtime.",
    },
    {
        "key": "social_support",
        "condition": lambda v: v <= 2,
        "icon": "🤝",
        "category": "Social Support",
        "body": "You rarely have someone to talk to when stressed.",
        "action": "Identify one trusted person — friend, family member, or counselor.",
    },
    {
        "key": "physical_activity",
        "condition": lambda v: v in ("Never", "1–2 days/week"),
        "icon": "🏃",
        "category": "Physical Activity",
        "body": "Low physical activity is linked to higher perceived stress.",
        "action": "Even 20 minutes of walking 3–4 times a week can help.",
    },
    {
        "key": "financial_stress",
        "condition": lambda v: v >= 4,
        "icon": "💼",
        "category": "Financial Stress",
        "body": "Financial concerns are weighing on you frequently.",
        "action": "Talk to your university's financial aid office; explore scholarships or part-time opportunities.",
    },
    {
        "key": "career_stress",
        "condition": lambda v: v >= 4,
        "icon": "🎯",
        "category": "Career Concerns",
        "body": "Uncertainty about the future is a common source of academic stress.",
        "action": "Visit your campus career services — planning reduces anxiety.",
    },
    {
        "key": "social_media",
        "condition": lambda v: v in ("5–6 hours", "More than 6 hours"),
        "icon": "📱",
        "category": "Social Media",
        "body": "Heavy social media use is linked to higher perceived stress.",
        "action": "Set a daily limit and keep phones out of study spaces.",
    },
    {
        "key": "assignment_stress",
        "condition": lambda v: v >= 4,
        "icon": "📚",
        "category": "Workload",
        "body": "Assignments and deadlines are frequently overwhelming you.",
        "action": "Break tasks into smaller pieces; try the Pomodoro method (25 min work, 5 min break).",
    },
    {
        "key": "academic_life_satisfaction",
        "condition": lambda v: v <= 2,
        "icon": "🎓",
        "category": "Academic Life",
        "body": "Your satisfaction with academic life is currently low.",
        "action": "Talk to a mentor or advisor about what could improve your day-to-day experience.",
    },
    {
        "key": "thought_break",
        "condition": lambda v: v in ("Often", "Very Often"),
        "icon": "🛑",
        "category": "Taking a Break",
        "body": "You have often thought about pausing your studies.",
        "action": "Speak with a counselor or mentor — options may exist that don't require giving up your degree.",
    },
    {
        "key": "screen_time",
        "condition": lambda v: v == "More than 8 hours",
        "icon": "🖥️",
        "category": "Screen Time",
        "body": "Over 8 hours of daily screen time can affect sleep and focus.",
        "action": "Break long screen sessions with short walks.",
    },
]


def generate_suggestions(user_input: dict, predicted_label: str) -> dict:
    """Build structured personalized suggestions with adaptive summary."""
    level = LEVEL_MESSAGES.get(predicted_label, LEVEL_MESSAGES["Moderate"])

    # ---- Collect triggered personalized tips ----
    tips = []
    for tip in FEATURE_TIPS:
        try:
            value = user_input.get(tip["key"])
            if value is not None and tip["condition"](value):
                tips.append({
                    "icon": tip["icon"],
                    "category": tip["category"],
                    "body": tip["body"],
                    "action": tip["action"],
                })
        except Exception:
            continue

    # ---- Adaptive note based on level + number of tips ----
    n_tips = len(tips)
    adaptive_note = ""

    if predicted_label == "Low":
        if n_tips == 0:
            adaptive_note = (
                " Nothing in your answers stands out as a concern — keep "
                "up your current routine."
            )
        elif n_tips <= 2:
            adaptive_note = (
                f" Your overall stress level appears low, but we noticed "
                f"{n_tips} area{'s' if n_tips != 1 else ''} that may be "
                f"worth improving — see the focus areas below."
            )
        else:
            adaptive_note = (
                f" Although the model predicts low stress, your answers "
                f"triggered {n_tips} areas that could be improved. Even if "
                f"you feel fine now, addressing them can prevent future stress."
            )

    elif predicted_label == "Moderate":
        if n_tips == 0:
            adaptive_note = (
                " No single area stands out strongly — small consistent "
                "habits may help you stay balanced."
            )
        elif n_tips <= 3:
            adaptive_note = (
                f" Your answers triggered {n_tips} focus area"
                f"{'s' if n_tips != 1 else ''}. Improving one of them often "
                f"creates momentum for the others."
            )
        else:
            adaptive_note = (
                f" Your answers triggered {n_tips} focus areas — addressing "
                f"several of these together can meaningfully reduce stress."
            )

    elif predicted_label == "High":
        if n_tips == 0:
            adaptive_note = (
                " Interestingly, no single area stands out in your answers, "
                "yet the overall pattern points to elevated stress. Consider "
                "talking with a counselor to explore what's going on."
            )
        elif n_tips <= 4:
            adaptive_note = (
                f" {n_tips} focus area{'s' if n_tips != 1 else ''} were "
                f"identified. Start with #01 — it usually has the biggest "
                f"impact."
            )
        else:
            adaptive_note = (
                f" {n_tips} focus areas were identified, which reinforces the "
                f"prediction. Please consider speaking to a counselor as well "
                f"as trying the suggestions below."
            )

    # ---- Universal habits ----
    always = [
        {
            "icon": "🧘",
            "title": "Breathing",
            "body": "Try 5 minutes of slow breathing (4 sec in, 6 sec out) "
                    "a few times a day to calm your nervous system.",
        },
        {
            "icon": "📅",
            "title": "Weekly Planning",
            "body": "A small weekly plan reduces decision fatigue and "
                    "last-minute rushing.",
        },
        {
            "icon": "👥",
            "title": "Stay Connected",
            "body": "Regular contact with friends and family, even brief, "
                    "protects against stress.",
        },
        {
            "icon": "🍽️",
            "title": "Eat & Rest Regularly",
            "body": "Skipping meals and irregular rest compound stress. "
                    "Simple routines help more than you'd think.",
        },
    ]

    return {
        "headline": level["headline"],
        "summary":  level["summary"] + adaptive_note,
        "tips":     tips,
        "always":   always,
    }