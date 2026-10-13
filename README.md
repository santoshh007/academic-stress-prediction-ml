# 🧠 Machine Learning-Based Prediction of Academic Stress Among Undergraduate Students in Nepal

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-red)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white)](https://sqlite.org/)

**BCA Final-Year Project — Tribhuvan University, Nepal**

**Authors:** Santosh Shrestha and Saroj Lamichhane  
**Supervisor:** Sulav Shrestha

This project investigates whether academic, lifestyle, demographic, and psychosocial factors can be used to predict academic stress among undergraduate students in Nepal.

The project uses survey data and machine learning to classify stress into three levels:

- **Low**
- **Moderate**
- **High**

It also includes SHAP-based model explanation, error analysis, a synthetic-data pipeline validation experiment, and a Streamlit application for demonstrating the prediction workflow.

> **Research note:** The genuine dataset currently contains only **39 responses**, so results from real data are preliminary. A separate 1,000-row synthetic dataset was used to validate the pipeline, not to make claims about real students.

## Table of Contents

- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Machine Learning](#machine-learning)
- [Synthetic Pipeline Validation](#synthetic-pipeline-validation)
- [SHAP Explainability](#shap-explainability)
- [Error Analysis](#error-analysis)
- [Streamlit Application](#streamlit-application)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Running the Project](#running-the-project)
- [Current Status](#current-status)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Authors](#authors)
- [License and Disclaimer](#license-and-disclaimer)

---

## Project Objectives

- Collect survey data from undergraduate students.
- Create an academic stress score from stress-related questions.
- Classify responses into Low, Moderate, and High stress levels.
- Explore relationships between survey features and the target.
- Train and compare different machine learning models.
- Evaluate model performance using cross-validation and held-out test data.
- Use SHAP to understand model predictions.
- Analyze prediction errors and potential overfitting.
- Validate the ML pipeline on a separate 1,000-row synthetic dataset.
- Build a Streamlit-based research prototype.
- Continue collecting genuine responses for future model improvement.

---

## Dataset

The questionnaire was created using Google Forms and contains questions covering several areas.

| Category | Examples |
|---|---|
| Demographics | Age, gender, university, program, year |
| Academic | CGPA, study hours, attendance, assignments, exams |
| Lifestyle | Sleep, social media, physical activity, screen time |
| Psychosocial | Financial stress, social support, career stress |
| Stress assessment | 10 questions using a 1–5 frequency scale |
| Other | Academic satisfaction, performance, biggest stress reason |

### Real Dataset (n=39)

- **39 genuine responses**
- **32 cleaned columns**
- **37 final ML features**
- Target classes: **Low / Moderate / High**

Four positively worded stress questions are reverse-scored before calculating the total stress score:

```text
stress_q4
stress_q5
stress_q7
stress_q8

reverse_score = 6 - original_score
```

The genuine dataset is too small to support reliable conclusions about the undergraduate population of Nepal.

### Synthetic Dataset (n=1000)

- **1,000 LLM-generated rows**, used for pipeline validation only.
- **35 columns**, with `Data_Source = "Synthetic"`.
- Uses the same reverse-scoring and tertile-split approach as the real-data pipeline.
- Is kept separate from the genuine-data workflow.
- **Is not treated as evidence about real students.**

See [`data/synthetic/README.md`](data/synthetic/README.md) for the synthetic dataset disclaimer.

---

## Project Workflow

```text
Survey Responses
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Stress Score + Target
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Cross-Validation
       ↓
Test Evaluation
       ↓
SHAP Explainability
       ↓
Error Analysis
       ↓
Streamlit Application
       ↓
Collect More Genuine Responses
```

---

## Machine Learning

The following models have been tested:

- Decision Tree
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost
- Support Vector Machine (SVM)
- Naive Bayes
- Dummy Classifier baseline

### Initial Real-Data Results

The initial model training was performed using the genuine dataset (`n=39`).

| Model | CV Accuracy | CV Macro F1 |
|---|---:|---:|
| Decision Tree | 0.605 ± 0.231 | 0.562 ± 0.262 |
| Logistic Regression | 0.414 ± 0.146 | 0.389 ± 0.161 |
| Random Forest | 0.410 ± 0.185 | 0.364 ± 0.197 |
| KNN | 0.381 ± 0.142 | 0.342 ± 0.163 |
| XGBoost | 0.381 ± 0.142 | 0.323 ± 0.159 |
| SVM | 0.376 ± 0.209 | 0.292 ± 0.223 |
| Naive Bayes | 0.286 ± 0.181 | 0.291 ± 0.181 |

**Selected model in the initial comparison:** The Decision Tree classifier achieved the highest cross-validation Macro F1 (**0.562**) among the seven models listed above.

Its held-out test performance was:

```text
Test Accuracy : 0.250
Test Macro F1 : 0.267
Baseline      : 0.386
```

The gap between cross-validation and held-out test performance suggests that the model does not generalize reliably with the current small dataset. The initial CV ranking should therefore not be interpreted as proof that the Decision Tree is the best model for the wider student population.

### Constrained Decision Tree Experiment

| Configuration | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| Original Decision Tree | 0.250 | 0.267 |
| Constrained (`max_depth=3`, `min_samples_leaf=3`) | **0.375** | **0.357** |

The constrained configuration performed better on the held-out test set in this experiment. Given the small sample size, this result is preliminary and needs to be re-evaluated with more genuine data.

---

## Synthetic Pipeline Validation

After reviewing the limitations of the 39-response dataset, we also created a separate **1,000-row synthetic dataset**. This experiment was designed to check the preprocessing and machine-learning workflow with more samples, not to generate research evidence about students.

### Real vs. Synthetic Results

| Model | Real Data CV Macro F1 | Synthetic Data CV Macro F1 |
|---|---:|---:|
| Random Forest | 0.364 | **0.744** |
| Logistic Regression | 0.389 | **0.737** |
| XGBoost | 0.323 | **0.736** |
| SVM | 0.292 | **0.724** |
| Decision Tree | **0.562** | **0.707** |
| Naive Bayes | 0.291 | **0.669** |
| KNN | 0.342 | **0.588** |
| Baseline | 0.184 | 0.170 |

These figures are the results recorded for the experiments and should be interpreted in the context of their respective datasets and evaluation setups. Better synthetic-data scores do **not** mean the model will achieve the same performance on genuine student responses.

The synthetic work is kept separate from the real-data workflow:

```text
data/synthetic/
notebooks_synthetic/
```

---

## SHAP Explainability

SHAP is used to inspect which features influence model predictions. The project generates plots such as:

```text
figures/
├── shap_importance.png
├── shap_beeswarm_*.png
├── shap_waterfall_*.png
├── confusion_matrix_test.png
├── cv_vs_test.png
├── real_vs_synthetic_comparison.png
└── synthetic_stress_distribution.png
```

### Top 7 Features by Mean |SHAP|

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `academic_life_satisfaction` | 0.164 |
| 2 | `social_support` | 0.145 |
| 3 | `gender_Male` | 0.120 |
| 4 | `academic_performance` | 0.107 |
| 5 | `cgpa` | 0.058 |
| 6 | `university_Sudurpaschim` | 0.037 |
| 7 | `physical_activity` | 0.032 |

These values describe feature contributions in the current analysis; they should not be interpreted as causal effects. SHAP findings from only 39 genuine responses may be unstable and require validation with a larger dataset.

---

## Error Analysis

The held-out test set contains only eight observations, so each prediction has a substantial effect on the reported metrics.

| Item | Result |
|---|---|
| Held-out observations | 8 |
| Correct predictions | 2 |
| Incorrect predictions | 6 |
| Error pattern | Errors were between adjacent classes (Low ↔ Moderate or Moderate ↔ High) |
| Low ↔ High errors | None observed |
| Mean confidence (correct predictions) | 1.000 |
| Mean confidence (incorrect predictions) | 0.945 |

**Key finding:** In this small test set, incorrect predictions had high reported confidence. This is consistent with a potential overfitting and overconfidence problem, but the very small test set limits how broadly this finding can be generalized.

---

## Streamlit Application

A fully implemented Streamlit application is included as a research prototype.

Current functionality includes:

- Interactive 21-input prediction form.
- Low / Moderate / High prediction using the trained Decision Tree.
- Visual stress-level scale.
- Personalized stress-reduction suggestions triggered by user answers.
- Priority-based focus-area cards.
- Nepal Mental Health Helpline (**1166**) prominently displayed.
- Research-limitation warnings on every screen.
- Optional anonymous response contribution.
- SQLite-based response storage, designed to avoid storing personal identifiers.

Application structure:

```text
app/
├── app.py
├── utils.py
├── style.py
├── db.py
├── pages/
└── collected_data/
    └── responses.db
```

The application is intended for research and demonstration purposes. It is **not a medical or psychological diagnostic tool**. Prediction outputs should not be used to make medical, treatment, or other high-stakes decisions.

---

## Project Structure

```text
academic-stress-prediction-ml/
│
├── .streamlit/
│   └── config.toml
├── .gitignore
│
├── app/
│   ├── app.py
│   ├── utils.py
│   ├── style.py
│   ├── db.py
│   ├── pages/
│   └── collected_data/
│       └── responses.db
│
├── data/
│   ├── raw/
│   │   └── student_stress.csv
│   ├── processed/
│   │   ├── student_stress_clean.csv
│   │   ├── student_stress_with_target.csv
│   │   ├── X_features.csv
│   │   └── y_target.csv
│   └── synthetic/
│       ├── README.md
│       ├── student_stress_synthetic_1000.csv
│       └── processed/
│           ├── student_stress_synthetic_clean.csv
│           ├── X_features_synthetic.csv
│           └── y_target_synthetic.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_explainability_shap.ipynb
│   └── 08_error_analysis.ipynb
│
├── notebooks_synthetic/
│   ├── 01_synth_data_understanding.ipynb
│   ├── 02_synth_cleaning_and_target.ipynb
│   ├── 03_synth_feature_engineering.ipynb
│   ├── 04_synth_model_training.ipynb
│   └── 05_real_vs_synthetic_comparison.ipynb
│
├── models/                     # Git-ignored; regenerable
│   ├── best_model.pkl
│   ├── best_model_metadata.json
│   ├── scaler.pkl
│   ├── best_model_synthetic.pkl
│   ├── best_model_synthetic_metadata.json
│   └── scaler_synthetic.pkl
│
├── reports/
│   ├── model_comparison.csv
│   ├── final_metrics.csv
│   └── model_comparison_synthetic.csv
│
├── figures/
│   ├── confusion_matrix_test.png
│   ├── cv_vs_test.png
│   ├── shap_importance.png
│   ├── shap_beeswarm_*.png
│   ├── shap_waterfall_*.png
│   ├── real_vs_synthetic_comparison.png
│   └── synthetic_stress_distribution.png
│
├── presentation/               # Project defense slides
├── regenerate_models.py
├── requirements.txt
└── README.md
```

> The tree documents the expected project layout. Confirm filenames and generated artifacts against the actual repository before relying on every listed path being present.

---

## Tech Stack

| Area | Technology |
|---|---|
| Programming | Python 3.12 |
| Environment | `uv` + virtual environment |
| Data processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Web application | Streamlit |
| Database | SQLite |
| Notebooks | Jupyter |
| Development | VS Code |
| Version control | Git and GitHub |

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/santoshh007/academic-stress-prediction-ml.git
cd academic-stress-prediction-ml
```

### 2. Create the environment

```bash
uv venv ml --python 3.12
```

Activate the environment.

**Linux/macOS**

```bash
source ml/bin/activate
```

**Windows PowerShell**

```powershell
.\ml\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use the appropriate activation command for your shell or configure your local execution policy according to your organization's guidance.

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Run the real-data notebooks

Run the notebooks in order:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### 5. Run the synthetic experiment

The synthetic workflow is maintained separately:

```text
notebooks_synthetic/
01 → 02 → 03 → 04 → 05
```

### 6. Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## Current Status

| Work | Status |
|---|---|
| Literature review | ✅ Completed |
| Proposal defense | ✅ Completed |
| Questionnaire | ✅ Completed |
| Initial genuine data collection | ✅ Completed |
| Data cleaning and EDA | ✅ Completed |
| Stress target construction | ✅ Completed |
| Feature engineering | ✅ Completed |
| Model training and evaluation | ✅ Completed |
| SHAP analysis | ✅ Completed |
| Error analysis | ✅ Completed |
| Streamlit prototype | ✅ Completed |
| Personalized suggestions | ✅ Completed |
| SQLite response storage | ✅ Completed |
| Synthetic pipeline validation (n=1000) | ✅ Completed |
| Larger genuine data collection | 🔄 Ongoing |
| Research paper | ⏳ Planned |
| Public deployment | ⏳ Planned |

---

## Limitations

- Only **39 genuine responses** are currently available.
- The current feature set contains **37 ML features**.
- The held-out test set contains only **8 samples**.
- The current model shows a generalization gap.
- Survey responses are self-reported.
- The sample is not representative of all undergraduate students in Nepal.
- External validation has not yet been performed.
- SHAP results may be unstable with the current sample.
- Synthetic data cannot replace genuine survey responses.
- The application is a research prototype, not a diagnostic instrument.

For these reasons, the current real-data results are **preliminary**.

---

## Future Work

The next major step is to collect **300–500 genuine responses** and repeat the analysis. Additional planned work includes:

1. Collect responses from more universities and academic programs.
2. Improve representation across regions and academic years.
3. Retrain and compare models using a larger genuine dataset.
4. Investigate feature selection and regularization.
5. Explore stress-score regression.
6. Study feature interactions using genuine data.
7. Perform external validation.
8. Continue improving the Streamlit application.
9. Prepare the research paper.

---

## Authors

**Santosh Shrestha** and **Saroj Lamichhane**  
BCA Final-Year Students  
Tribhuvan University, Nepal

**Supervisor:** Sulav Shrestha

GitHub repository: [santoshh007/academic-stress-prediction-ml](https://github.com/santoshh007/academic-stress-prediction-ml)

---

## License and Disclaimer

This project is intended for **academic and research purposes**. No specific open-source license is stated here; add a `LICENSE` file if the project is to be distributed under a particular license.

The current model is a research prototype and must not be used for medical diagnosis, treatment decisions, or other high-stakes decisions.

> **Project status:** Research prototype — Machine Learning + SHAP + Streamlit + Synthetic Pipeline Validation
