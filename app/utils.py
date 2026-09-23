"""
utils.py — Shared helpers for the Streamlit app.

Loads the fitted model and scaler, and provides a single function to
transform a dict of user responses into the 37-feature vector the model
expects.
"""

import os
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Resolve paths relative to this file (works regardless of cwd)
# ---------------------------------------------------------------
APP_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(APP_DIR)

MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
SCALER_PATH   = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "X_features.csv")

# ---------------------------------------------------------------
# Load model and scaler once at import time
# ---------------------------------------------------------------
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# The exact feature order used during training
FEATURE_COLUMNS = pd.read_csv(FEATURES_PATH, nrows=0).columns.tolist()

# ---------------------------------------------------------------
# Ordinal mappings (must match Notebook 04)
# ---------------------------------------------------------------
ORDINAL_MAPPINGS = {
    "age": {
        "17-19": 1, "20-22": 2, "23-25": 3, "Above 25": 4,
    },
    "year": {
        "1st Year": 1, "2nd Year": 2, "3rd Year": 3, "4th Year": 4, "5th Year": 5,
    },
    "cgpa": {
        "Not Available Yet": 0, "2.0–2.49": 1, "2.5–2.99": 2, "3.0–3.49": 3, "3.5–4.0": 4,
    },
    "study_hours": {
        "Less than 1": 1, "1–2": 2, "3–4": 3, "5–6": 4, "More than 6 hours": 5,
    },
    "attendance": {
        "Below 60%": 1, "60–70%": 2, "71–80%": 3, "81–90%": 4, "Above 90%": 5,
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
        "Less than 3 hours": 1, "3–5 hours": 2, "6–8 hours": 3, "More than 8 hours": 4,
    },
    "thought_break": {
        "Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Very Often": 5,
    },
}

# Nominal categories seen during training (drop_first=True in notebook 04
# means one category was dropped per nominal feature — we must reproduce that)
NOMINAL_CATEGORIES = {
    "gender":       ["Female", "Male", "Prefer not to say"],
    "university":   ["Kathmandu University", "Other", "Pokhara University",
                     "Purbanchal University", "Rajarshi Janak University",
                     "Sudurpaschim University", "Tribhuvan University"],
    "program":      ["BBS", "BCA", "BDS", "BE/B.Tech", "BHM", "BIM",
                     "BSc CSIT", "BSc Nursing", "Law(LLB)", "MBBS"],
    "part_time_job": ["No", "Yes"],
}


def _one_hot(base_name, value, categories, drop_first=True):
    """
    Reproduce sklearn/pandas get_dummies(drop_first=True) behavior.
    Returns a dict of {feature_name: 0/1}.
    """
    cats = categories[1:] if drop_first else categories
    return {f"{base_name}_{c}": (1 if value == c else 0) for c in cats}


def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    Convert a dict of raw user responses into a DataFrame with the
    exact 37 columns the model expects.

    user_input keys (all strings or ints):
        age, gender, university, program, year, cgpa, study_hours,
        attendance, assignment_stress, exam_count, sleep_hours,
        social_media, physical_activity, part_time_job, screen_time,
        financial_stress, social_support, academic_life_satisfaction,
        thought_break, career_stress, academic_performance
    """
    row = {}

    # Ordinal: map strings to ints
    for col, mapping in ORDINAL_MAPPINGS.items():
        row[col] = mapping[user_input[col]]

    # Numeric 1-5 scale features (already ints)
    for col in [
        "assignment_stress", "financial_stress", "social_support",
        "academic_life_satisfaction", "career_stress", "academic_performance",
    ]:
        row[col] = int(user_input[col])

    # One-hot encoded nominals
    row.update(_one_hot("gender",        user_input["gender"],        NOMINAL_CATEGORIES["gender"]))
    row.update(_one_hot("university",    user_input["university"],    NOMINAL_CATEGORIES["university"]))
    row.update(_one_hot("program",       user_input["program"],       NOMINAL_CATEGORIES["program"]))
    row.update(_one_hot("part_time_job", user_input["part_time_job"], NOMINAL_CATEGORIES["part_time_job"]))

    # Reorder to match the exact training column order
    df = pd.DataFrame([row])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


def predict(user_input: dict):
    """
    Take raw user responses -> predicted label + class probabilities.

    Returns:
        (label: str, proba_dict: {class: probability})
    """
    X = preprocess_input(user_input)
    X_scaled = scaler.transform(X)

    label = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    proba_dict = {cls: float(p) for cls, p in zip(model.classes_, proba)}

    return label, proba_dict