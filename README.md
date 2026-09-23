# 🧠 Machine Learning-Based Prediction of Academic Stress Among Undergraduate Students in Nepal

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-red)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite\&logoColor=white)](https://sqlite.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)](#-project-status)

> **BCA Final-Year Project | Tribhuvan University, Nepal**

A survey-based machine learning research project that investigates whether demographic, academic, lifestyle, and psychosocial factors can be used to predict **academic stress levels among undergraduate students in Nepal**.

The system classifies responses into **Low, Moderate, and High** stress levels and includes machine learning, SHAP-based explainability, error analysis, and an interactive Streamlit research prototype.

> ⚠️ **Research Note:** The current validated dataset contains only **39 genuine responses**. The present model is therefore a **research prototype**, not a validated or clinically reliable stress-assessment system.

---

## 🎯 Objectives

* Collect and analyze undergraduate student survey data.
* Construct an academic stress score from 10 stress-related questions.
* Categorize responses into **Low / Moderate / High** stress levels.
* Perform exploratory data analysis and feature engineering.
* Compare multiple machine learning algorithms.
* Evaluate models using cross-validation and held-out testing.
* Investigate model behavior using **SHAP**.
* Perform detailed prediction error analysis.
* Provide an interactive Streamlit research prototype.
* Collect anonymous responses through the application for future research.

---

## 🔬 Research Principle

> **The data determines which features matter — not the researcher.**

The project does not assume that a particular demographic, academic, or lifestyle factor causes academic stress.

Model results and feature explanations are treated as **exploratory evidence that requires validation**, especially because the current dataset is small.

---

## 📊 Dataset

Data are collected through the **Academic Stress Survey for Undergraduate Students** using Google Forms.

| Property               |        Current Status |
| ---------------------- | --------------------: |
| Genuine responses      |                **39** |
| Cleaned observations   |                **39** |
| Cleaned variables      |                **32** |
| Final ML features      |                **37** |
| Target                 | Low / Moderate / High |
| Collection method      |      Anonymous survey |
| Planned larger dataset | **300–500 responses** |

### Main Feature Groups

* 👤 Demographics — age, gender, university, program, year, CGPA
* 📚 Academic — study hours, attendance, assignment stress, exams
* 🛌 Lifestyle — sleep, social media, physical activity, part-time job, screen time
* 🧠 Psychosocial — financial stress, social support, career stress, academic satisfaction and performance
* 📋 Stress assessment — 10 questions using a 1–5 frequency scale

Four positively worded stress questions are reverse-scored:

```text
stress_q4
stress_q5
stress_q7
stress_q8

reverse_score = 6 - original_score
```

---

## 🔄 Research Workflow

```text
Google Forms Survey
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Stress Score Construction
        ↓
Low / Moderate / High Target
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Model Training
        ↓
Cross-Validation + Testing
        ↓
SHAP Explainability
        ↓
Error Analysis
        ↓
Streamlit Research Prototype
        ↓
Future Data Collection & Validation
```

---

## 🤖 Machine Learning

The project currently compares:

* Decision Tree
* Logistic Regression
* Random Forest
* K-Nearest Neighbors
* XGBoost
* Support Vector Machine
* Naive Bayes
* Dummy Classifier baseline

### Current Results

The initial dataset contains **37 features and only 39 observations**, creating a high risk of overfitting.

| Model               |       CV Accuracy |       CV Macro F1 |
| ------------------- | ----------------: | ----------------: |
| **Decision Tree**   | **0.605 ± 0.231** | **0.562 ± 0.262** |
| Logistic Regression |     0.414 ± 0.146 |     0.389 ± 0.161 |
| Random Forest       |     0.410 ± 0.185 |     0.364 ± 0.197 |
| KNN                 |     0.381 ± 0.142 |     0.342 ± 0.163 |
| XGBoost             |     0.381 ± 0.142 |     0.323 ± 0.159 |
| SVM                 |     0.376 ± 0.209 |     0.292 ± 0.223 |
| Naive Bayes         |     0.286 ± 0.181 |     0.291 ± 0.181 |

However, on the held-out test set:

```text
Test Accuracy : 0.250
Test Macro F1 : 0.267
Baseline      : 0.386
```

This indicates that the current model **does not generalize reliably** and should not be presented as a validated prediction system.

---

## 🧠 Explainable AI

The project uses **SHAP TreeExplainer** to investigate model behavior.

The current prototype generates:

```text
figures/
├── shap_importance.png
├── shap_beeswarm_*.png
└── shap_waterfall_*.png
```

Because of the very small dataset and observed overfitting, SHAP feature rankings are treated as **unstable exploratory results**, not established psychological relationships.

---

## 🌐 Streamlit Web Application

A Streamlit research prototype has been implemented.

### Features

* Interactive prediction form
* Low / Moderate / High prediction
* Visual stress-level indicator
* Personalized stress-reduction suggestions
* Feature-based recommendations
* Priority/focus-area cards
* Research limitation warnings
* Optional anonymous research contribution
* SQLite response storage

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

> ⚠️ The application is a **research prototype** and is not a diagnostic or mental-health assessment system.

---

## 🗄️ Anonymous Response Database

Anonymous responses contributed through the application are stored locally using SQLite.

```text
app/collected_data/responses.db
```

The database is excluded from version control.

It stores survey inputs, prediction information, and a timestamp without intentionally collecting personal identifiers.

---

## 📁 Project Structure

```text
academic-stress-prediction-ml/
│
├── app/
│   ├── app.py
│   ├── utils.py
│   ├── style.py
│   ├── db.py
│   ├── pages/
│   └── collected_data/
│
├── data/
│   ├── raw/
│   └── processed/
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
├── models/
├── reports/
├── figures/
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Category         | Technology               |
| ---------------- | ------------------------ |
| Language         | Python 3.12              |
| Environment      | uv + virtual environment |
| Data Processing  | pandas, NumPy            |
| Visualization    | Matplotlib, Seaborn      |
| Machine Learning | scikit-learn, XGBoost    |
| Explainable AI   | SHAP                     |
| Web Application  | Streamlit                |
| Database         | SQLite                   |
| Notebooks        | Jupyter                  |
| IDE              | VS Code                  |
| Version Control  | Git + GitHub             |

---

## 🚀 Reproducibility

### 1. Clone

```bash
git clone https://github.com/iamsaroj2058/Academic-stress-prediction-ml.git
cd Academic-stress-prediction-ml
```

### 2. Create Environment

```bash
uv venv ml --python 3.12
```

Linux/macOS:

```bash
source ml/bin/activate
```

Windows:

```powershell
ml\Scripts\activate
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Run Notebooks

Run in order:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### 5. Run Streamlit

```bash
streamlit run app/app.py
```

---

## 🚧 Project Status

| Task                       | Status        |
| -------------------------- | ------------- |
| Literature review          | ✅ Completed   |
| Proposal defense           | ✅ Completed   |
| Questionnaire design       | ✅ Completed   |
| Initial data collection    | ✅ Completed   |
| Data cleaning              | ✅ Completed   |
| EDA                        | ✅ Completed   |
| Stress-target construction | ✅ Completed   |
| Feature engineering        | ✅ Completed   |
| Model training             | ✅ Completed   |
| Model evaluation           | ✅ Completed   |
| SHAP explainability        | ✅ Completed   |
| Error analysis             | ✅ Completed   |
| Streamlit prototype        | ✅ Implemented |
| Personalized suggestions   | ✅ Completed   |
| SQLite response storage    | ✅ Completed   |
| Larger data collection     | 🔄 Ongoing    |
| Research paper             | ⏳ Planned     |
| Public deployment          | ⏳ Planned     |

---

## ⚠️ Limitations

The current results should be interpreted cautiously because:

* The genuine dataset currently contains only **39 responses**.
* There are **37 ML features**, resulting in a high feature-to-sample ratio.
* The held-out test set contains only **8 observations**.
* The model shows a substantial cross-validation/test performance gap.
* Survey responses are self-reported.
* The current sample is not representative of all undergraduate students in Nepal.
* No external validation has been performed.
* SHAP explanations are unstable with the current sample size.

Therefore:

> **This project is a methodological research prototype, not a validated academic-stress prediction system.**

---

## 🔮 Future Work

1. Collect **300–500 genuine responses**.
2. Improve representation across universities, programs, regions, and academic years.
3. Re-train and re-evaluate models using the larger dataset.
4. Investigate feature selection and regularization.
5. Explore continuous stress-score regression.
6. Investigate feature interactions empirically.
7. Perform external validation using another dataset or student cohort.
8. Continue improving the Streamlit research application.
9. Prepare the project for research-paper publication.

---

## 🔐 Ethical Considerations

* Participation is voluntary.
* Survey responses are intended to remain anonymous.
* Data are collected for academic/research purposes.
* The application does not provide a medical or psychological diagnosis.
* Predictions should not be used for high-stakes decisions.
* Model limitations are communicated to users.
* Future data collection will continue to follow appropriate consent and privacy practices.

---

## 👨‍💻 Author

**Saroj Lamichhane**

BCA (Bachelor of Computer Application) Final-Year Student
Tribhuvan University, Nepal

🔗 **GitHub:**
https://github.com/iamsaroj2058/Academic-stress-prediction-ml

---

## 📄 License

This project is intended for **academic and research use**.

The current model is a research prototype and must not be used for:

* Mental-health diagnosis
* Medical assessment
* Treatment decisions
* High-stakes decisions about students

Any reuse of the project should respect applicable privacy, consent, and research-ethics requirements.

---

### ⭐ Project Status

> **Research Prototype — Machine Learning + Explainable AI + Streamlit**

The complete initial research pipeline has been implemented, including data processing, machine learning, evaluation, SHAP explainability, error analysis, personalized recommendations, and anonymous SQLite-based response collection. The next major research phase is **larger genuine data collection and model re-evaluation**.
