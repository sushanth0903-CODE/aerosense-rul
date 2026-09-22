# AeroSense ✈️

### NASA C-MAPSS Turbofan Remaining Useful Life Prediction using Random Forest

AeroSense is a predictive maintenance project that uses **Random Forest Regression** to estimate the **Remaining Useful Life (RUL)** of simulated turbofan engines from sensor telemetry.

The project is built using NASA's **C-MAPSS FD001** dataset and includes a complete machine-learning pipeline:

**data preparation → RUL calculation → engine-wise validation → Random Forest training → evaluation → model saving → interactive Streamlit dashboard**

Built as a first-year machine learning project with an emphasis on understanding the complete implementation rather than treating the model as a black box.

---

## 1. Problem Statement

Aircraft engines are monitored using many sensors during operation. As an engine degrades, its sensor measurements can change.

Instead of waiting for an engine to fail, predictive maintenance attempts to estimate:

> **How many operating cycles does the engine have remaining before failure?**

This quantity is called **Remaining Useful Life (RUL)**.

For example:

```text
Current cycle = 140
Failure cycle = 200

RUL = 200 - 140
    = 60 cycles
```

RUL prediction is a **regression problem** because the model predicts a numerical quantity rather than a category such as healthy/faulty.

---

# 2. Dataset

AeroSense uses the **NASA C-MAPSS FD001** turbofan engine degradation dataset.

FD001 contains simulated run-to-failure trajectories for turbofan engines under a single operating condition and a single degradation mode.

The raw FD001 data contains:

* Engine / unit number
* Operating cycle
* 3 operational settings
* 21 sensor measurements

This gives a total of **26 columns before adding the target RUL column**.

The training data contains complete engine trajectories up to failure.

The test data contains partial trajectories, together with a separate file containing the final RUL value for each test engine.

### NASA Dataset

Official NASA dataset:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

---

# 3. Why FD001?

C-MAPSS contains multiple subsets:

```text
FD001
FD002
FD003
FD004
```

AeroSense currently uses **FD001** as the project scope.

FD001 is useful for learning the complete machine-learning workflow without immediately introducing multiple operating conditions and multiple fault modes.

The project can later be extended to FD002–FD004, but the current implementation intentionally focuses on one well-defined dataset.

---

# 4. Data Availability and GitHub

The raw NASA dataset is **not committed to this repository**.

The GitHub repository contains the source code required to obtain and process the dataset.

When the project runs, the pipeline looks for:

```text
data/raw/
```

and expects:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

The pipeline can also download the C-MAPSS archive automatically when these files are missing.

If automatic downloading is unavailable, the dataset can be downloaded manually and the required FD001 files can be placed inside:

```text
data/raw/
```

### Why is the dataset not stored in GitHub?

Keeping the raw dataset outside the repository keeps the Git repository lightweight and avoids redistributing the complete dataset.

The `.gitignore` file explicitly excludes:

```text
data/raw/*
artifacts/*
reports/*
```

while allowing generated folders to be used locally.

---

# 5. Understanding the RUL Target

## Training data

For each training engine:

```text
RUL = Maximum cycle of that engine - Current cycle
```

Example:

```text
Maximum cycle = 200

Current cycle:
20  → RUL = 180
50  → RUL = 150
100 → RUL = 100
150 → RUL = 50
190 → RUL = 10
200 → RUL = 0
```

AeroSense caps the training target at:

```text
MAX_RUL = 125
```

Therefore, very large early-life RUL values are clipped to 125.

---

## Test data

NASA provides one final RUL value for each test engine.

For every row in a test trajectory, AeroSense reconstructs the corresponding RUL using:

```text
RUL =
Final observed test cycle
- Current cycle
+ NASA final RUL
```

For the final benchmark-style test evaluation, AeroSense uses the **last observed row of each test engine**.

This matches the point at which the supplied test RUL corresponds to the engine's latest observation.

---

# 6. Machine Learning Approach

AeroSense uses:

```text
Random Forest Regressor
```

Random Forest is an ensemble machine-learning algorithm composed of many decision trees.

Each decision tree learns relationships such as:

```text
if sensor_3 < threshold
    go left
else
    go right
```

Many trees are trained and their predictions are combined to produce the final prediction.

For regression, the forest effectively combines the outputs of the individual trees.

---

# 7. Why Random Forest?

Random Forest was selected because it works well with numerical tabular data and is straightforward to understand and explain.

Important characteristics relevant to AeroSense:

* Captures non-linear relationships.
* Can model interactions between different sensor measurements.
* Requires relatively little preprocessing.
* Does **not require feature scaling or normalization**.
* Provides feature-importance values.
* Works naturally as a regression model for numerical RUL prediction.

Unlike many algorithms, the model can work directly with the sensor values without first transforming every feature onto the same numerical scale.

---

# 8. Data Processing Pipeline

The complete pipeline is:

```text
NASA C-MAPSS FD001
        │
        ▼
Download / load raw files
        │
        ▼
Parse space-separated telemetry
        │
        ▼
Create training RUL
        │
        ▼
Reconstruct test RUL
        │
        ▼
Split training data by engine
        │
        ▼
Remove zero / near-zero variance features
        │
        ▼
Train Random Forest Regressor
        │
        ├──────────────► Feature Importance
        │
        ▼
Validation Evaluation
        │
        ▼
Final Test Evaluation
        │
        ▼
Save Model + Metadata + Reports
        │
        ▼
Streamlit Dashboard
```

---

# 9. Feature Preparation

The raw dataset contains:

```text
1  Engine ID
1  Cycle
3  Operational settings
21 Sensor measurements
```

The model does **not** use the engine ID as a feature.

Why?

Because engine ID is only an identifier:

```text
Engine 1
Engine 2
Engine 3
...
```

It does not represent a physical measurement of the engine.

The model therefore uses the telemetry-related numerical features and excludes:

```text
unit_number
RUL
```

from the model inputs.

Features whose variance is effectively zero are also removed automatically.

This is determined from the training data rather than relying on a permanently hard-coded list.

---

# 10. Data Leakage Prevention

One important problem in time-series/entity-based datasets is **data leakage**.

A random row split could produce something like:

```text
Engine 17, cycle 20  → training
Engine 17, cycle 21  → validation
Engine 17, cycle 22  → training
Engine 17, cycle 23  → validation
```

The model would then see parts of the same engine trajectory in both sets.

That can make validation performance misleadingly optimistic.

AeroSense instead performs the split at the **engine level** using:

```python
GroupShuffleSplit
```

Therefore, a complete engine trajectory belongs to either:

```text
Training
```

or:

```text
Validation
```

but not both.

This makes the validation procedure more meaningful for the problem.

---

# 11. Random Forest Configuration

The current baseline configuration is:

```text
n_estimators      = 100
max_depth         = 15
min_samples_split = 5
random_state      = 42
n_jobs             = -1
oob_score         = True
```

### Parameter meaning

**`n_estimators = 100`**

Creates 100 decision trees.

**`max_depth = 15`**

Limits the maximum depth of each tree.

**`min_samples_split = 5`**

A node must contain at least five samples before it can be split.

**`random_state = 42`**

Makes the experiment reproducible.

**`n_jobs = -1`**

Allows scikit-learn to use all available CPU cores for training.

**`oob_score = True`**

Enables out-of-bag scoring during Random Forest training.

---

# 12. Evaluation Metrics

AeroSense reports four metrics.

## RMSE

**Root Mean Squared Error**

RMSE gives greater importance to larger prediction errors.

The unit is:

```text
cycles
```

A lower RMSE indicates smaller prediction error.

---

## MAE

**Mean Absolute Error**

MAE represents the average absolute difference between the predicted RUL and actual RUL.

Example:

```text
Actual RUL      = 50
Predicted RUL   = 43

Absolute error  = 7 cycles
```

MAE is easy to interpret because it is expressed directly in cycles.

---

## R²

**Coefficient of Determination**

R² measures how well the model explains variation in the target relative to a baseline based on the target mean.

A value closer to 1 generally indicates stronger explanatory performance on the evaluated data.

---

## NASA / PHM Asymmetric Score

RUL prediction does not treat every type of mistake equally.

AeroSense therefore also calculates the asymmetric score used for the C-MAPSS / PHM-style evaluation.

Let:

```text
d = predicted_RUL - actual_RUL
```

Then:

```text
if d < 0:

    score = exp(-d / 13) - 1

else:

    score = exp(d / 10) - 1
```

The individual penalties are summed across the evaluated engines.

```text
Lower score → better
Perfect prediction → 0
```

The asymmetric structure reflects the fact that early and late RUL errors are penalized differently.

---

# 13. Current Test Result

The current trained FD001 model produced the following final-cycle test evaluation:

| Metric     |        Result |
| ---------- | ------------: |
| RMSE       | 17.660 cycles |
| MAE        | 13.161 cycles |
| R²         |         0.819 |
| NASA Score |       498.498 |

These are the results from the current project run and are provided as a reproducible project result, not as a claim of certified aviation performance.

---

# 14. Feature Importance

Random Forest provides feature-importance values.

AeroSense saves these values and uses them to identify the most influential telemetry features in the trained model.

The model then selects the **five most important sensor features** for the interactive dashboard.

This allows the project to answer a useful question:

> **Which sensor measurements contributed most to the model's predictions?**

The feature importance report is generated locally as:

```text
reports/feature_importance.csv
```

---

# 15. Streamlit Dashboard

AeroSense includes an interactive Streamlit application.

Launch it with:

```bash
python -m streamlit run app/streamlit_app.py
```

Or on Windows:

```powershell
py -m streamlit run app/streamlit_app.py
```

The dashboard provides:

### Engine selection

Choose a test engine by ID.

### Latest telemetry

The application starts with the engine's latest available observed telemetry.

### Sensor controls

The five most important sensor values can be modified using sliders.

### Live RUL prediction

The Random Forest model immediately predicts RUL using the modified telemetry.

### RUL gauge

The dashboard visually displays:

```text
Predicted Remaining Useful Life
```

in operating cycles.

### Feature importance

The top ten model features are displayed so the user can inspect which inputs contribute most to the model.

### Telemetry snapshot

The dashboard also shows the selected engine's current cycle and important sensor values.

---

# 16. Dashboard Status Categories

The dashboard uses three educational categories:

| Predicted RUL | Dashboard status    |
| ------------: | ------------------- |
|   > 60 cycles | SAFE                |
|  21–60 cycles | MAINTENANCE WARNING |
|   ≤ 20 cycles | CRITICAL            |

These are **demo categories created for the project interface**.

They are not certified aviation maintenance limits and should not be interpreted as operational recommendations.

---

# 17. Project Structure

The **actual GitHub repository** currently contains:

```text
aerosense-rul/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── evaluate.py
│   └── model_train.py
│
├── tests/
│   ├── test_data_pipeline.py
│   └── test_evaluate.py
│
├── .gitignore
├── PROJECT_NOTES.md
├── README.md
└── requirements.txt
```

### Generated locally

The following directories are generated or populated during execution and are intentionally not committed:

```text
data/
├── raw/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── CMAPSSData.zip
│
artifacts/
├── model.joblib
└── metadata.joblib
│
reports/
├── metrics.json
├── feature_importance.csv
└── test_predictions.csv
```

These files can be reproduced by running the project.

---

# 18. What Each Source File Does

## `src/data_pipeline.py`

Responsible for the data layer.

It:

* downloads the NASA C-MAPSS archive when required;
* extracts the FD001 files;
* reads the space-separated data;
* assigns column names;
* calculates training RUL;
* reconstructs test RUL;
* performs engine-wise train/validation splitting;
* identifies informative features.

---

## `src/evaluate.py`

Contains evaluation utilities.

It implements:

* RMSE;
* MAE;
* R²;
* NASA asymmetric score;
* combined metric generation;
* JSON metric saving;
* dashboard health-status logic.

---

## `src/model_train.py`

Contains the main training pipeline.

It:

1. loads FD001;
2. creates the engine-wise validation split;
3. selects model features;
4. removes near-zero variance features;
5. trains the Random Forest;
6. evaluates the validation set;
7. evaluates the final cycle of each test engine;
8. calculates feature importance;
9. saves the trained model;
10. saves metadata;
11. writes evaluation reports.

---

## `app/streamlit_app.py`

Contains the interactive dashboard.

It:

* loads the saved model;
* loads model metadata;
* loads FD001 telemetry;
* lets the user select a test engine;
* provides sensor controls;
* predicts RUL;
* displays the RUL gauge;
* shows feature importance;
* displays telemetry information.

---

## `tests/test_data_pipeline.py`

Tests important RUL data-processing behaviour.

---

## `tests/test_evaluate.py`

Tests the evaluation functions, including the asymmetric NASA score and regression metrics.

---

## `PROJECT_NOTES.md`

Contains the important design decisions made during the project, including:

* why feature scaling is not used;
* why engine-wise splitting is used;
* why raw data is not committed;
* how final-cycle testing is performed;
* why dashboard thresholds are only educational.

---

# 19. Installation

## Prerequisites

For the current dependency pins, use a Python environment in the **3.10–3.13 range**.

Clone the repository:

```bash
git clone https://github.com/sushanth0903-CODE/aerosense-rul.git
cd aerosense-rul
```

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

---

# 20. Getting the Dataset

The training pipeline can attempt to download the C-MAPSS archive automatically.

Run:

```bash
python -m src.model_train
```

If the automatic download does not work, download the NASA C-MAPSS dataset manually and place the archive or required FD001 files in:

```text
data/raw/
```

The required FD001 files are:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

---

# 21. Train the Model

From the repository root:

```bash
python -m src.model_train
```

On Windows:

```powershell
py -m src.model_train
```

The training process:

```text
Load FD001
   ↓
Prepare RUL
   ↓
Split by engine
   ↓
Train Random Forest
   ↓
Validate
   ↓
Evaluate final test cycles
   ↓
Save model and reports
```

Generated files include:

```text
artifacts/model.joblib
artifacts/metadata.joblib

reports/metrics.json
reports/feature_importance.csv
reports/test_predictions.csv
```

These generated files are ignored by Git.

---

# 22. Run the Dashboard

The dashboard requires the model artifacts and reports produced by training.

Run:

```bash
python -m streamlit run app/streamlit_app.py
```

Windows:

```powershell
py -m streamlit run app/streamlit_app.py
```

Then open the local Streamlit address shown in the terminal.

---

# 23. Run the Tests

The project contains automated tests in:

```text
tests/
```

Run:

```bash
python -m pytest
```

On Windows:

```powershell
py -m pytest
```

The test suite currently covers the main data-processing and evaluation utilities.

---

# 24. Reproducibility

Several settings are fixed to make experiments reproducible:

```text
random_state = 42
```

and the Random Forest configuration is explicitly defined in:

```text
src/model_train.py
```

The project also saves model metadata containing:

* selected features;
* top sensor features;
* sensor descriptions;
* feature ranges;
* training RUL cap;
* Random Forest parameters.

---

# 25. Important Project Limitations

AeroSense is an educational machine-learning project, not an aircraft-engine maintenance system.

Important limitations include:

### Simulated data

C-MAPSS is a simulated engine degradation dataset rather than live aircraft telemetry.

### Single subset

Only FD001 is currently implemented.

### Single baseline model

The project currently focuses on Random Forest Regression rather than comparing many algorithms.

### Demo status thresholds

The Safe / Warning / Critical categories are UI demonstrations.

### No operational deployment

The model should not be used for real aviation maintenance decisions.

---

# 26. Possible Future Extensions

The current project can be expanded in several directions:

```text
FD001
  ↓
FD002 / FD003 / FD004
```

Possible future improvements include:

* comparison with XGBoost;
* Gradient Boosting;
* Extra Trees;
* hyperparameter tuning;
* richer time-series features;
* sliding-window features;
* trend and degradation analysis;
* uncertainty estimation;
* model comparison;
* explainability with SHAP;
* experiment tracking;
* CI testing;
* deployment of the dashboard;
* support for additional C-MAPSS operating conditions.

These are extensions rather than requirements of the current implementation.

---

# 27. Reproducing the Current Result

A complete local run is:

```bash
git clone https://github.com/sushanth0903-CODE/aerosense-rul.git
cd aerosense-rul

python -m venv .venv

# Activate the environment

python -m pip install -r requirements.txt

python -m src.model_train

python -m streamlit run app/streamlit_app.py
```

For Windows Python Launcher:

```powershell
py -m venv .venv
.venv\Scripts\activate

py -m pip install -r requirements.txt

py -m src.model_train

py -m streamlit run app/streamlit_app.py
```

---

# 28. References

1. NASA Open Data — **CMAPSS Jet Engine Simulated Data**

   https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

2. Saxena, A., Goebel, K., Simon, D., & Eklund, N.
   **Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation**, PHM08.

3. Scikit-learn documentation — **RandomForestRegressor**

   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

4. Streamlit documentation

   https://docs.streamlit.io/

---

## Project Summary

```text
AeroSense
│
├── Problem
│   └── Predict Remaining Useful Life
│
├── Dataset
│   └── NASA C-MAPSS FD001
│
├── ML Task
│   └── Regression
│
├── Model
│   └── Random Forest Regressor
│
├── Validation
│   └── Engine-wise GroupShuffleSplit
│
├── Metrics
│   ├── RMSE
│   ├── MAE
│   ├── R²
│   └── NASA asymmetric score
│
├── Outputs
│   ├── Trained model
│   ├── Metadata
│   ├── Feature importance
│   └── Test predictions
│
└── Interface
    └── Interactive Streamlit dashboard
```

**AeroSense demonstrates a complete end-to-end machine-learning workflow: from raw turbofan telemetry to an interpretable RUL prediction and an interactive application.**
