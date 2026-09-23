# 🧠 Machine Learning-Based Prediction of Academic Stress Among Undergraduate Students in Nepal

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-red)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter\&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)](#project-status)
[![License](https://img.shields.io/badge/License-Academic%20%26%20Research%20Use-lightgrey)](#license)

> **BCA Final-Year Project | Tribhuvan University, Nepal**

A survey-based machine learning research prototype for investigating whether demographic, academic, lifestyle, and psychosocial factors can be used to predict **academic stress levels among undergraduate students in Nepal**.

**⚠️ Research Status:** This project is currently a **prototype** based on an initial sample of **39 responses (n = 39)**. The current dataset is too small to support reliable or generalizable machine-learning conclusions. The results documented here are therefore treated as an evaluation of the research pipeline rather than evidence of a deployable academic-stress prediction system.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Research Objectives](#-research-objectives)
* [Research Principle](#-research-principle)
* [Dataset](#-dataset)
* [Research Workflow](#-research-workflow)
* [Project Structure](#-project-structure)
* [Technology Stack](#-technology-stack)
* [Implementation](#-implementation)
* [Modeling Results](#-modeling-results)
* [Key Findings](#-key-findings)
* [Explainable AI](#-explainable-ai)
* [Error Analysis](#-error-analysis)
* [Web Application](#-web-application)
* [Reproducibility](#-reproducibility)
* [Project Status](#-project-status)
* [Limitations](#-limitations)
* [Future Work](#-future-work)
* [Ethical and Research Considerations](#-ethical-and-research-considerations)
* [Author](#-author)
* [License](#-license)

---

# 🔎 Overview

Academic stress is a relevant issue among university students and may be influenced by multiple academic, lifestyle, demographic, and psychosocial factors. This project investigates the feasibility of using **machine learning classification techniques** to categorize undergraduate students into three stress-level groups: **Low, Moderate, and High**.

Data were collected through a voluntary and anonymous Google Forms questionnaire designed for undergraduate students. The research pipeline covers data understanding, cleaning, exploratory analysis, stress-score construction, feature engineering, model training, cross-validation, held-out testing, SHAP-based explainability, and error analysis.

The initial dataset contains only **39 participants**, with **37 model features**, resulting in a feature-to-sample ratio of approximately **p/n ≈ 0.95**. Consequently, the current models exhibit clear signs of overfitting and poor generalization. The project therefore treats the present results as a **research prototype and methodological experiment**, not as evidence that academic stress can currently be predicted reliably in Nepalese undergraduate students.

The primary next step is to expand the dataset to approximately **300–500 responses** before drawing stronger conclusions or developing a public-facing predictive system.

---

# 🎯 Research Objectives

## General Objective

To investigate the feasibility of using machine learning techniques to predict academic stress levels among undergraduate students in Nepal using survey-derived demographic, academic, lifestyle, and psychosocial features.

## Specific Objectives

1. Collect structured survey data from undergraduate students.
2. Clean and preprocess the collected survey responses.
3. Construct a stress score from validated-style frequency-based stress questions.
4. Categorize students into Low, Moderate, and High stress groups.
5. Perform exploratory data analysis to understand the collected dataset.
6. Engineer suitable numerical and categorical machine-learning features.
7. Train and compare multiple classification algorithms.
8. Evaluate models using stratified cross-validation and a held-out test set.
9. Investigate model explanations using SHAP.
10. Analyze prediction errors and model confidence.
11. Develop a Streamlit-based research prototype for interactive prediction.
12. Establish a reproducible workflow that can be extended when a larger dataset is collected.

---

# 🧪 Research Principle

> **The data determines which features matter — not the researcher.**

This project intentionally avoids assuming that a particular demographic, academic, or lifestyle variable must be an important predictor.

Feature importance and model explanations are treated as **empirical outputs that require validation**, rather than conclusions that are assumed in advance.

Particular care is taken to distinguish:

* correlation from causation,
* model behavior from real-world relationships,
* exploratory findings from validated findings,
* cross-validation performance from independent test performance,
* and prototype results from generalizable research conclusions.

---

# 📊 Dataset

## Data Collection

The initial dataset was collected using a Google Forms questionnaire titled:

**Academic Stress Survey for Undergraduate Students**

Participation was:

* Voluntary
* Anonymous
* Approximately 5–7 minutes
* Intended for academic/research purposes

### Current Dataset

| Property                        |                 Value |
| ------------------------------- | --------------------: |
| Initial responses               |                **39** |
| Cleaned observations            |                **39** |
| Cleaned columns                 |                **32** |
| Final model features            |                **37** |
| Target classes                  | Low / Moderate / High |
| Current feature-to-sample ratio |        p/n ≈ **0.95** |
| Planned larger dataset          | **300–500 responses** |

> ⚠️ **Important:** The current sample size is insufficient for reliable machine-learning generalization. The 39-response dataset is primarily being used to validate the research workflow and identify methodological issues before larger-scale data collection.

---

## Survey Variables

### Demographic Features

* Age
* Gender
* University
* Program
* Year
* CGPA

### Academic Features

* Study hours
* Attendance
* Assignment stress
* Exam count

### Lifestyle Features

* Sleep hours
* Social media usage
* Physical activity
* Part-time job
* Screen time

### Psychosocial Features

* Financial stress
* Social support
* Career stress
* Academic life satisfaction
* Academic performance
* Thought break

### Stress Assessment

The questionnaire contains 10 stress-related questions:

```text
stress_q1
stress_q2
...
stress_q10
```

Each question uses a five-point frequency scale:

```text
1 → Never
2 → Rarely
3 → Sometimes
4 → Often
5 → Very Often
```

Four positively worded questions were reverse-scored:

```text
stress_q4
stress_q5
stress_q7
stress_q8
```

using:

```text
reverse_score = 6 - original_score
```

The corrected values were then used to calculate the total stress score.

---

# 🔬 Research Workflow

```text
┌──────────────────────────────┐
│      Google Forms Survey     │
│          n = 39              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Data Understanding      │
│   Shape / Types / Missing    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Data Cleaning          │
│ Rename / Remove / Validate   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Exploratory Data Analysis    │
│ Distributions / Stress Score │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Target Construction     │
│     Low / Moderate / High    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Feature Engineering      │
│ Ordinal + One-Hot Encoding   │
│ Remove Leakage / Free Text   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Train/Test Split (80/20)  │
│      Stratified Sampling     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Model Training         │
│  7 Classification Algorithms │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stratified 5-Fold CV        │
│ Accuracy + Macro F1          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Held-Out Test Evaluation  │
│      n = 8 test samples      │
└──────────────┬───────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐ ┌──────────────┐
│ SHAP / XAI   │ │ Error        │
│ Explainability│ │ Analysis     │
└──────────────┘ └──────────────┘
        │             │
        └──────┬──────┘
               ▼
┌──────────────────────────────┐
│   Research Discussion &      │
│   Limitations / Future Work  │
└──────────────────────────────┘
```

---

# 📁 Project Structure

```text
academic-stress-prediction-ml/
│
├── app/                              # Streamlit research prototype
│   ├── app.py                        # Main application
│   ├── utils.py                      # Encoding & prediction utilities
│   ├── pages/                        # Additional application pages
│   └── collected_data/               # Research response storage
│
├── data/
│   ├── raw/
│   │   └── student_stress.csv        # Raw survey dataset
│   │
│   └── processed/
│       ├── student_stress_clean.csv
│       ├── student_stress_with_target.csv
│       ├── X_features.csv
│       └── y_target.csv
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
├── models/                           # Gitignored; regenerable
│   ├── best_model.pkl
│   ├── best_model_metadata.json
│   └── scaler.pkl
│
├── reports/
│   ├── model_comparison.csv
│   └── final_metrics.csv
│
├── figures/
│   ├── confusion_matrix_test.png
│   ├── cv_vs_test.png
│   ├── shap_importance.png
│   ├── shap_beeswarm_*.png
│   └── shap_waterfall_*.png
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technology Stack

| Category             | Technology                 | Purpose                            |
| -------------------- | -------------------------- | ---------------------------------- |
| Programming Language | Python 3.12                | Core development                   |
| Environment          | `uv` + virtual environment | Dependency/environment management  |
| Data Processing      | pandas                     | Data loading and manipulation      |
| Numerical Computing  | NumPy                      | Numerical operations               |
| Visualization        | Matplotlib                 | Research figures                   |
| Visualization        | Seaborn                    | Statistical visualization          |
| Machine Learning     | scikit-learn               | Preprocessing, models, evaluation  |
| Gradient Boosting    | XGBoost                    | Comparative ML model               |
| Explainable AI       | SHAP                       | Model explanation                  |
| Imbalanced Learning  | imbalanced-learn           | Future class-balancing experiments |
| Web Application      | Streamlit                  | Interactive research prototype     |
| Notebooks            | Jupyter                    | Experimentation and analysis       |
| IDE                  | VS Code                    | Development environment            |
| Data Export          | openpyxl                   | Spreadsheet-related data handling  |
| Version Control      | Git                        | Source control                     |
| Repository           | GitHub                     | Project hosting                    |

---

# 🧩 Implementation

The research pipeline is organized into eight notebooks.

## 01 — Data Understanding

`01_data_understanding.ipynb`

The raw CSV was inspected to understand:

* Dataset dimensions
* Column names
* Data types
* Missing values
* Survey structure
* Initial data quality

Initial raw dataset:

```text
39 rows × 34 columns
```

---

## 02 — Data Cleaning

`02_data_cleaning.ipynb`

The cleaning process included:

1. Removing the timestamp column.
2. Removing the consent column.
3. Renaming columns using `snake_case`.
4. Checking for missing values.
5. Validating the resulting dataset.

Output:

```text
39 rows × 32 columns
```

Saved as:

```text
data/processed/student_stress_clean.csv
```

---

## 03 — Exploratory Data Analysis

`03_exploratory_data_analysis.ipynb`

The analysis included:

* Age distribution
* Gender distribution
* University distribution
* Program distribution
* Stress-question distributions
* Stress-score distribution
* Target-class distribution

### Stress Score

After reverse-scoring the positively worded questions, the total stress score was calculated as:

```text
stress_score = sum(correctly_oriented_stress_questions)
```

The theoretical range was:

```text
10 – 50
```

Observed statistics:

| Statistic          | Value |
| ------------------ | ----: |
| Minimum            |    10 |
| Maximum            |    39 |
| Mean               | 28.95 |
| Standard deviation |  4.82 |

### Target Construction

Stress levels were constructed using the 33rd and 67th percentiles:

| Class     | Samples |
| --------- | ------: |
| Low       |      15 |
| Moderate  |      13 |
| High      |      11 |
| **Total** |  **39** |

Output:

```text
data/processed/student_stress_with_target.csv
```

---

# ⚙️ Feature Engineering

`04_feature_engineering.ipynb`

Potentially leaky variables were removed before model training.

### Excluded From Features

```text
stress_q1 ... stress_q10
reversed stress questions
stress_score
stress_level
biggest_stress_reason
```

The stress questions and derived stress score were excluded because they directly contribute to the target definition and could introduce target leakage.

The free-text field was excluded because NLP processing was outside the scope of the current prototype.

### Ordinal Encoding

The following ordered variables were ordinal encoded:

```text
age
year
cgpa
study_hours
attendance
exam_count
sleep_hours
social_media
physical_activity
screen_time
thought_break
```

### One-Hot Encoding

Nominal variables were one-hot encoded:

```text
gender
university
program
part_time_job
```

with:

```text
drop_first=True
```

Final feature matrix:

```text
39 samples × 37 features
```

This resulted in:

```text
p/n ≈ 37 / 39 ≈ 0.95
```

which is a major limitation for the current dataset.

---

# 🤖 Model Training

`05_model_training.ipynb`

The dataset was divided using a stratified 80/20 split:

```text
Training samples: 31
Testing samples:   8
```

Stratification was used to preserve class proportions as much as possible.

A `StandardScaler` was fitted **only on the training data** and subsequently applied to the test data to avoid test-set information influencing preprocessing.

Seven classification algorithms were evaluated:

* Decision Tree
* Logistic Regression
* Random Forest
* K-Nearest Neighbors
* XGBoost
* Support Vector Machine
* Naive Bayes

A `DummyClassifier` was also used as a baseline.

---

# 📈 Modeling Results

## Cross-Validation Results

The following results are from stratified 5-fold cross-validation on the training set.

| Model                    |          Accuracy |          Macro F1 |
| ------------------------ | ----------------: | ----------------: |
| **Decision Tree**        | **0.605 ± 0.231** | **0.562 ± 0.262** |
| Logistic Regression      |     0.414 ± 0.146 |     0.389 ± 0.161 |
| Random Forest            |     0.410 ± 0.185 |     0.364 ± 0.197 |
| KNN (k=5)                |     0.381 ± 0.142 |     0.342 ± 0.163 |
| XGBoost                  |     0.381 ± 0.142 |     0.323 ± 0.159 |
| SVM (RBF)                |     0.376 ± 0.209 |     0.292 ± 0.223 |
| Naive Bayes              |     0.286 ± 0.181 |     0.291 ± 0.181 |
| DummyClassifier baseline |             0.386 |             0.184 |

The Decision Tree had the highest cross-validation macro F1 among the evaluated models and was therefore selected for further investigation.

> **Important:** The cross-validation result should not be interpreted as evidence of reliable real-world performance because the training dataset contains only 31 observations.

---

# 🧪 Held-Out Test Results

The selected Decision Tree was evaluated on the independent held-out test set:

```text
Test samples = 8
```

| Metric            |    Result |
| ----------------- | --------: |
| Test Accuracy     | **0.250** |
| Test Macro F1     | **0.267** |
| Baseline Accuracy | **0.386** |

The model's test performance was **below the baseline accuracy**, demonstrating poor generalization on the held-out data.

This creates a substantial gap between cross-validation and held-out test performance:

```text
Cross-validation Macro F1 : 0.562
Held-out Test Macro F1   : 0.267
```

This gap is consistent with the project's small-sample and high-dimensional setting and provides evidence that the initial model is overfitting.

---

# 🔍 Key Findings

The current analysis supports the following observations.

### 1. Severe Small-Sample Limitation

The dataset contains:

```text
n = 39
p = 37
```

Therefore:

```text
p/n ≈ 0.95
```

The number of model features is almost as large as the number of observations.

This creates a high risk of overfitting and unstable estimates.

---

### 2. Large Generalization Gap

The Decision Tree achieved:

```text
CV Macro F1   = 0.562
Test Macro F1 = 0.267
```

The substantial difference indicates that the cross-validation result does not translate reliably to unseen observations.

---

### 3. Test Performance Was Below Baseline

The Decision Tree achieved:

```text
Accuracy = 0.250
```

while the DummyClassifier baseline achieved:

```text
Accuracy = 0.386
```

Therefore, the current Decision Tree should **not** be presented as a reliable predictive model.

---

### 4. Errors Were Concentrated Between Adjacent Classes

The six incorrect predictions were all between neighboring stress categories:

```text
Low       ↔ Moderate
Moderate  ↔ High
```

There were no:

```text
Low ↔ High
```

confusions in the held-out test set.

This observation is exploratory and should be reassessed using a substantially larger dataset.

---

### 5. Model Confidence Was Excessive

The Decision Tree showed:

| Prediction Type       | Mean Confidence |
| --------------------- | --------------: |
| Correct predictions   |           1.000 |
| Incorrect predictions |           0.945 |

The combination of high confidence and poor accuracy indicates severe overconfidence in the current tree.

This behavior is consistent with a highly flexible tree trained on a very small dataset.

---

### 6. A Constrained Tree Reduced Overfitting

An additional experiment constrained the tree using:

```python
max_depth=3
min_samples_leaf=3
```

Results on the held-out test set:

| Model                     | Accuracy | Macro F1 |
| ------------------------- | -------: | -------: |
| Original Decision Tree    |    0.250 |    0.267 |
| Constrained Decision Tree |    0.375 |    0.357 |

Although these results remain unreliable because the test set contains only eight observations, the experiment provides additional evidence that unrestricted tree complexity contributed to overfitting.

---

# 🧠 Explainable AI with SHAP

`07_explainability_shap.ipynb`

SHAP's `TreeExplainer` was used to investigate how the selected Decision Tree behaved internally.

The largest observed mean absolute SHAP values were:

| Rank | Feature                      | Mean |SHAP| |
| ---: | ---------------------------- | ----------: |
|    1 | `academic_life_satisfaction` |       0.164 |
|    2 | `social_support`             |       0.145 |
|    3 | `gender_Male`                |       0.120 |
|    4 | `academic_performance`       |       0.107 |
|    5 | `cgpa`                       |       0.058 |
|    6 | `university_Sudurpaschim`    |       0.037 |
|    7 | `physical_activity`          |       0.032 |

Approximately 30 other features had zero mean absolute SHAP contribution in the observed model.

### SHAP Visualizations

The project generates:

```text
figures/
├── shap_importance.png
├── shap_beeswarm_*.png
└── shap_waterfall_*.png
```

These include:

* Global feature importance
* Class-specific beeswarm plots
* Individual prediction waterfall plots

### Important Interpretation

Some observed SHAP directions contradicted the expected theoretical interpretation.

For example, higher `academic_life_satisfaction` was associated with higher predicted stress in some model explanations.

This should **not** be interpreted as evidence that academic-life satisfaction causes higher stress.

Instead, given the extremely small dataset and poor held-out performance, such counterintuitive relationships are treated as evidence that the current feature-importance estimates are unstable and potentially driven by overfitting.

---

# 🧾 Error Analysis

`08_error_analysis.ipynb`

The eight test observations were examined individually.

```text
Correct predictions : 2 / 8
Incorrect predictions: 6 / 8
```

Therefore:

```text
Test accuracy = 25%
```

The error analysis examined:

* Actual class
* Predicted class
* Prediction confidence
* Class distance
* Misclassification patterns

### Main observation

All six errors were adjacent-class errors:

```text
Low ↔ Moderate
Moderate ↔ High
```

No:

```text
Low ↔ High
```

errors occurred.

Because the test set contains only eight observations, this pattern should be treated as an exploratory observation rather than a population-level conclusion.

---

# 🌐 Web Application

A Streamlit-based research prototype is currently under development.

## Current Structure

```text
app/
├── app.py
├── utils.py
├── pages/
└── collected_data/
```

`utils.py` contains the preprocessing and prediction utilities required to reproduce the model's feature encoding.

## Planned Features

The application is intended to provide:

* Interactive prediction form
* Approximately 21 user inputs
* Low / Moderate / High predicted category
* Prominent research-model limitation warning
* Optional anonymous contribution of responses
* About page
* Methodology page
* Research information

### Important Safety/Research Notice

The application is a **research prototype**.

It is **not a diagnostic system**, mental-health assessment tool, or substitute for professional support.

Because the current model was developed using only 39 observations and performs poorly on the held-out test set, its predictions should not be interpreted as reliable assessments of an individual's actual stress level.

---

# 🔁 Reproducibility

The project uses `uv` for Python environment and dependency management.

## 1. Clone the Repository

```bash
git clone https://github.com/iamsaroj2058/Academic-stress-prediction-ml.git

cd Academic-stress-prediction-ml
```

## 2. Create the Python Environment

```bash
uv venv ml --python 3.12
```

Activate it on Linux/macOS:

```bash
source ml/bin/activate
```

On Windows:

```powershell
ml\Scripts\activate
```

## 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## 4. Run the Notebooks

Run the notebooks in sequence:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### Pipeline

```text
01_data_understanding
        ↓
02_data_cleaning
        ↓
03_exploratory_data_analysis
        ↓
04_feature_engineering
        ↓
05_model_training
        ↓
06_model_evaluation
        ↓
07_explainability_shap
        ↓
08_error_analysis
```

Each stage generates outputs used by subsequent stages.

---

# ▶️ Running the Streamlit Prototype

After installing the dependencies:

```bash
streamlit run app/app.py
```

The application will start a local Streamlit server.

The application is currently intended for research/development use rather than public deployment.

---

# 📦 Generated Outputs

The notebooks generate processed datasets, trained model artifacts, reports, and research figures.

### Processed Data

```text
data/processed/
├── student_stress_clean.csv
├── student_stress_with_target.csv
├── X_features.csv
└── y_target.csv
```

### Model Artifacts

```text
models/
├── best_model.pkl
├── best_model_metadata.json
└── scaler.pkl
```

The `models/` directory is gitignored because the model artifacts are regenerable from the notebooks.

### Reports

```text
reports/
├── model_comparison.csv
└── final_metrics.csv
```

### Figures

```text
figures/
├── confusion_matrix_test.png
├── cv_vs_test.png
├── shap_importance.png
├── shap_beeswarm_*.png
└── shap_waterfall_*.png
```

---

# 🚧 Project Status

| Component                  | Status         |
| -------------------------- | -------------- |
| Literature review          | ✅ Completed    |
| Proposal defense           | ✅ Completed    |
| Questionnaire design       | ✅ Completed    |
| Initial data collection    | ✅ Completed    |
| Project structure          | ✅ Completed    |
| Python environment         | ✅ Completed    |
| Data cleaning              | ✅ Completed    |
| Exploratory data analysis  | ✅ Completed    |
| Stress-target construction | ✅ Completed    |
| Feature engineering        | ✅ Completed    |
| Model training             | ✅ Completed    |
| Model comparison           | ✅ Completed    |
| Model evaluation           | ✅ Completed    |
| SHAP explainability        | ✅ Completed    |
| Error analysis             | ✅ Completed    |
| Research discussion        | ✅ Completed    |
| Larger data collection     | 🔄 Ongoing     |
| Streamlit application      | 🔄 In Progress |
| Research paper             | ⏳ Planned      |

---

# ⚠️ Limitations

The current results have substantial limitations.

## 1. Small Sample Size

The dataset contains only:

```text
n = 39
```

This is insufficient for developing a reliable predictive model with 37 features.

---

## 2. High Feature-to-Sample Ratio

The final feature matrix contains:

```text
37 features / 39 observations
```

giving:

```text
p/n ≈ 0.95
```

This creates a high risk of overfitting.

---

## 3. Very Small Test Set

Only eight observations were allocated to the held-out test set:

```text
80% training → 31
20% testing  → 8
```

Consequently, a single prediction changes the test accuracy by:

```text
1 / 8 = 12.5 percentage points
```

This makes the test metrics highly unstable.

---

## 4. Class Distribution

The target classes contain:

```text
Low       = 15
Moderate  = 13
High      = 11
```

While the overall distribution is not extremely imbalanced, the small absolute number of observations per class becomes problematic after the train/test split.

---

## 5. Self-Reported Data

Survey responses are self-reported and may be affected by:

* Recall bias
* Response bias
* Interpretation differences
* Social desirability
* Individual reporting differences

---

## 6. Limited Population Representation

The current sample should not be considered representative of all undergraduate students in Nepal.

The sample size and collection strategy do not support population-level generalization.

---

## 7. No External Validation

The current model has not been evaluated on:

* A separate institution
* A separate geographic sample
* A separate dataset
* A future cohort

External validation remains necessary.

---

## 8. Unstable Explainability

SHAP results are highly sensitive to the trained model.

Given the current sample size and overfitting behavior, the observed SHAP rankings should not be treated as stable estimates of real-world feature importance.

---

## 9. Prototype-Level Results

The current model should be considered:

> **A methodological prototype, not a validated academic-stress prediction system.**

---

# 🚀 Future Work

The primary objective for the next research phase is to increase the sample size and improve methodological robustness.

## 1. Expand Data Collection

Target:

```text
300–500 responses
```

This is the most important next step before making stronger modeling claims.

---

## 2. Improve Sample Diversity

Future collection should attempt to include students across:

* Different regions
* Universities
* University types
* Academic programs
* Academic years

This would improve the diversity of the dataset and allow subgroup analyses.

---

## 3. Investigate Alternative Target Formulations

Instead of only using categorical stress levels, future work could investigate:

```text
Continuous stress-score regression
```

This may preserve more information than converting a continuous score into three categories.

---

## 4. Feature Reduction

With a larger dataset, investigate:

* Feature selection
* Regularization
* Dimensionality reduction
* Stability-based feature selection

---

## 5. Interaction Analysis

Potential interactions can be investigated empirically, for example:

```text
financial stress × part-time job
study hours × sleep hours
academic workload × career stress
social support × academic performance
```

These relationships should be tested rather than assumed.

---

## 6. External Validation

A future model should ideally be evaluated using data from:

* Another institution
* Another geographic region
* A later student cohort

This would provide stronger evidence about generalization.

---

## 7. Improve the Web Application

The Streamlit application can eventually support:

* Research data collection
* Model prediction
* Methodology information
* Model limitations
* Dataset monitoring
* Research dashboards

Any public deployment should retain clear warnings about the model's research status and limitations.

---

## 8. Research Publication

After collecting a sufficiently large and diverse dataset, the project can be extended into a research paper covering:

```text
Literature Review
       ↓
Research Methodology
       ↓
Data Collection
       ↓
Statistical Analysis
       ↓
Machine Learning
       ↓
Explainable AI
       ↓
Validation
       ↓
Discussion
       ↓
Limitations
       ↓
Future Research
```

---

# 🔐 Ethical and Research Considerations

Because the project concerns academic stress, responsible research practices are important.

The current project follows these principles:

* Participation is voluntary.
* Survey responses are intended to be anonymous.
* Data should be used for academic/research purposes.
* Personally identifying information should not be collected unnecessarily.
* Model outputs should not be interpreted as medical or psychological diagnoses.
* Predictions should not be used to make high-stakes decisions about students.
* Model limitations should be clearly communicated.
* Future data collection should maintain appropriate consent and privacy practices.

---

# 📚 Research Pipeline Summary

The complete system can be summarized as:

```text
                 ┌─────────────────────┐
                 │  Student Survey     │
                 │     Responses        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Cleaning       │
                 │ & Validation        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Stress Score        │
                 │ Construction        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Target Classes      │
                 │ Low / Mod / High    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Engineering │
                 │ Encoding / Leakage  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ ML Model Training   │
                 │ 7 Algorithms        │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Cross-Validation + Testing │
              └─────────────┬──────────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        ┌─────────────────┐    ┌─────────────────┐
        │ SHAP Explainable│    │ Error Analysis  │
        │ AI              │    │                 │
        └────────┬────────┘    └────────┬────────┘
                 │                      │
                 └──────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Research Findings   │
                 │ & Limitations       │
                 └─────────────────────┘
```

---

# 👨‍💻 Author

**Santosh Shrestha and Saroj Lamichhane**

BCA (Bachelor of Computer Application)
Tribhuvan University, Nepal

GitHub:
https://github.com/santoshh007/academic-stress-prediction-ml

This project was developed as a **BCA final-year academic project** and is intended to serve as a foundation for further research into machine-learning approaches to academic stress among undergraduate students in Nepal.

---

# 📄 License

This project is intended for **academic and research use**.

The current machine-learning model is a research prototype and **must not be used for actual mental-health assessment, diagnosis, treatment decisions, or other high-stakes decisions**.

Any reuse of the dataset, model, or research findings should appropriately acknowledge the original project and respect applicable privacy, consent, and research-ethics requirements.

---

## ⭐ Project Status Summary

> **Current status: Research Prototype**

The complete machine-learning research pipeline has been implemented from data understanding through error analysis and explainability. However, the current dataset contains only **39 observations**, and the held-out evaluation demonstrates poor generalization.

The project therefore prioritizes **methodological transparency over inflated performance claims**.

The next major milestone is to collect a substantially larger dataset, reassess the modeling pipeline, validate the results on unseen populations, and determine whether the observed relationships remain stable.

> **The current results are evidence about this prototype and dataset—not evidence about the academic-stress patterns of undergraduate students in Nepal as a whole.**
