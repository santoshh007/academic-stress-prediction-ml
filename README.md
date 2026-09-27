# Machine Learning-Based Prediction of Academic Stress Among Undergraduate Students in Nepal

**BCA Final-Year Project — Tribhuvan University, Nepal**

This project looks at whether academic, lifestyle, demographic, and psychosocial factors can be used to predict academic stress among undergraduate students in Nepal.

The project uses survey data and machine learning to classify stress into three levels:

* **Low**
* **Moderate**
* **High**

It also includes SHAP-based model explanation, error analysis, and a Streamlit application for testing the prediction workflow.

> **Note:** The current genuine dataset has only **39 responses**, so the real-data results are considered preliminary. A separate synthetic-data experiment was also carried out to test the pipeline with a larger dataset, but the synthetic results are not treated as real student findings.

---

## Project Objectives

* Collect survey data from undergraduate students.
* Create an academic stress score from the stress-related questions.
* Classify responses into Low, Moderate, and High stress levels.
* Explore the relationship between different survey features and the target.
* Train and compare different machine learning models.
* Evaluate model performance using cross-validation and test data.
* Use SHAP to understand model predictions.
* Analyze prediction errors.
* Build a Streamlit-based research prototype.
* Continue collecting genuine responses for future model improvement.

---

## Dataset

The questionnaire was created using Google Forms and contains questions covering several areas.

| Category          | Examples                                                  |
| ----------------- | --------------------------------------------------------- |
| Demographics      | Age, gender, university, program, year                    |
| Academic          | CGPA, study hours, attendance, assignments, exams         |
| Lifestyle         | Sleep, social media, physical activity, screen time       |
| Psychosocial      | Financial stress, social support, career stress           |
| Stress assessment | 10 questions using a 1–5 frequency scale                  |
| Other             | Academic satisfaction, performance, biggest stress reason |

### Current Real Dataset

* **39 genuine responses**
* **32 cleaned columns**
* **37 final ML features**
* Target: **Low / Moderate / High**

Four positively worded stress questions are reverse-scored before calculating the total stress score:

```text
stress_q4
stress_q5
stress_q7
stress_q8

reverse_score = 6 - original_score
```

The current dataset is too small to make reliable claims about the undergraduate population of Nepal.

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
Larger Genuine Dataset
```

---

## Machine Learning

The following models have been tested:

* Decision Tree
* Logistic Regression
* Random Forest
* KNN
* XGBoost
* SVM
* Naive Bayes
* Dummy Classifier baseline

### Initial Results

The initial model training was performed using the genuine dataset (`n=39`).

| Model               |   CV Accuracy |   CV Macro F1 |
| ------------------- | ------------: | ------------: |
| Decision Tree       | 0.605 ± 0.231 | 0.562 ± 0.262 |
| Logistic Regression | 0.414 ± 0.146 | 0.389 ± 0.161 |
| Random Forest       | 0.410 ± 0.185 | 0.364 ± 0.197 |
| KNN                 | 0.381 ± 0.142 | 0.342 ± 0.163 |
| XGBoost             | 0.381 ± 0.142 | 0.323 ± 0.159 |
| SVM                 | 0.376 ± 0.209 | 0.292 ± 0.223 |
| Naive Bayes         | 0.286 ± 0.181 | 0.291 ± 0.181 |

The Decision Tree had the highest cross-validation Macro F1 in the initial experiment.

However, its held-out test performance was:

```text
Test Accuracy : 0.250
Test Macro F1 : 0.267
Baseline      : 0.386
```

This large difference between cross-validation and test performance is one of the main reasons the current model should not be considered reliable yet.

---

## Synthetic Pipeline Experiment

After seeing the limitations of the 39-response dataset, I also created a separate **1,000-row synthetic dataset**.

The purpose of this experiment was not to create research evidence about students. It was mainly to check how the existing preprocessing and ML pipeline behaves when there are substantially more samples.

### Results

| Model               | Real Data | Synthetic Data |
| ------------------- | --------: | -------------: |
| Random Forest       |     0.364 |      **0.744** |
| Logistic Regression |     0.389 |      **0.737** |
| XGBoost             |     0.323 |      **0.736** |
| SVM                 |     0.292 |      **0.724** |
| Decision Tree       |     0.562 |      **0.707** |
| Naive Bayes         |     0.291 |      **0.669** |
| KNN                 |     0.342 |      **0.588** |
| Baseline            |     0.184 |          0.170 |

These results show that the pipeline can produce much more stable results when working with a larger dataset under the synthetic-data conditions.

However, **synthetic performance should not be interpreted as expected performance on real students**. The next step is therefore to collect more genuine survey responses and run the same pipeline again.

The synthetic work is kept separate from the real-data workflow:

```text
data/synthetic/
notebooks_synthetic/
```

---

## SHAP Explainability

SHAP is used to inspect which features are influencing individual model predictions.

The project generates:

```text
figures/
├── shap_importance.png
├── shap_beeswarm_*.png
├── shap_waterfall_*.png
├── confusion_matrix_test.png
├── cv_vs_test.png
└── real_vs_synthetic_comparison.png
```

The SHAP results from the current 39-response dataset are treated cautiously because the sample is too small for stable conclusions.

---

## Streamlit Application

A Streamlit application is being developed as a research prototype.

Current functionality includes:

* Survey input form
* Stress prediction
* Low / Moderate / High result
* Personalized suggestions
* Feature-based recommendations
* Research limitation information
* Optional anonymous response contribution
* SQLite-based response storage

Project structure:

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

The application is intended for research and demonstration purposes. It is **not a medical or psychological diagnostic tool**.

---

## Project Structure

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
│   ├── processed/
│   └── synthetic/
│       ├── README.md
│       ├── student_stress_synthetic_1000.csv
│       └── processed/
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
├── models/
├── reports/
├── figures/
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Area             | Technology               |
| ---------------- | ------------------------ |
| Programming      | Python 3.12              |
| Environment      | uv + virtual environment |
| Data             | pandas, NumPy            |
| Visualization    | Matplotlib, Seaborn      |
| Machine Learning | scikit-learn, XGBoost    |
| Explainability   | SHAP                     |
| Web App          | Streamlit                |
| Database         | SQLite                   |
| Notebooks        | Jupyter                  |
| Development      | VS Code                  |
| Version Control  | Git + GitHub             |

---

## Running the Project

### Clone the repository

```bash
git clone https://github.com/iamsaroj2058/Academic-stress-prediction-ml.git
cd Academic-stress-prediction-ml
```

### Create the environment

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

### Install dependencies

```bash
uv pip install -r requirements.txt
```

### Run the notebooks

Run the real-data notebooks in order:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

The synthetic experiment is kept separately:

```text
notebooks_synthetic/
01 → 02 → 03 → 04 → 05
```

### Run the Streamlit app

```bash
streamlit run app/app.py
```

---

## Current Status

| Work                            | Status     |
| ------------------------------- | ---------- |
| Literature review               | ✅          |
| Proposal defense                | ✅          |
| Questionnaire                   | ✅          |
| Initial genuine data collection | ✅          |
| Data cleaning                   | ✅          |
| EDA                             | ✅          |
| Stress target construction      | ✅          |
| Feature engineering             | ✅          |
| Model training                  | ✅          |
| Model evaluation                | ✅          |
| SHAP analysis                   | ✅          |
| Error analysis                  | ✅          |
| Streamlit prototype             | ✅          |
| Personalized suggestions        | ✅          |
| SQLite response storage         | ✅          |
| Synthetic pipeline experiment   | ✅          |
| Larger genuine data collection  | 🔄 Ongoing |
| Research paper                  | ⏳ Planned  |
| Public deployment               | ⏳ Planned  |

---

## Limitations

The main limitation at the moment is the size of the genuine dataset.

* Only **39 genuine responses** are currently available.
* There are **37 ML features**.
* The test set contains only **8 samples**.
* The current model shows a noticeable generalization gap.
* Survey responses are self-reported.
* The sample is not representative of all undergraduate students in Nepal.
* External validation has not yet been performed.
* SHAP results may be unstable with the current sample.
* Synthetic data cannot replace genuine survey responses.

Because of these limitations, the current results are considered **preliminary**.

---

## Future Work

The main next step is to collect **300–500 genuine responses** and repeat the complete analysis.

Other planned work includes:

1. Collect more responses from different universities and programs.
2. Improve representation across regions and academic years.
3. Re-train and compare the models using the larger real dataset.
4. Investigate feature selection and regularization.
5. Explore stress-score regression.
6. Study feature interactions using genuine data.
7. Perform external validation.
8. Continue improving the Streamlit application.
9. Prepare the research paper.

---

## Author

**Saroj Lamichhane**

BCA Final-Year Student
Tribhuvan University, Nepal

GitHub:
https://github.com/iamsaroj2058/Academic-stress-prediction-ml

---

## License

This project is intended for **academic and research purposes**.

The current model is a research prototype and should not be used for medical diagnosis, treatment decisions, or other high-stakes decisions.

---

> **Project status:** Research prototype — Machine Learning + SHAP + Streamlit + Synthetic Pipeline Validation
