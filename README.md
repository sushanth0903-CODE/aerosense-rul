# AeroSense ✈️

## NASA C-MAPSS Turbofan Remaining Useful Life Prediction using Random Forest

AeroSense is a predictive-maintenance machine-learning project that uses **Random Forest Regression** to estimate the **Remaining Useful Life (RUL)** of simulated turbofan engines from sensor telemetry.

The project uses NASA's **C-MAPSS FD001** dataset and implements a complete machine-learning workflow:

**data preparation → RUL calculation → engine-wise validation → Random Forest training → evaluation → model saving → interactive Streamlit dashboard**

AeroSense was developed as a first-year machine-learning project with an emphasis on understanding the complete implementation rather than treating the model as a black box.

---

## 1. Problem Statement

Aircraft engines are monitored using multiple sensors during operation. As an engine degrades, its sensor measurements can change.

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

RUL prediction is a **regression problem** because the model predicts a numerical quantity rather than a category such as healthy or faulty.

---

# 2. Dataset

AeroSense uses the **NASA C-MAPSS Jet Engine Simulated Data**, specifically the **FD001** subset.

### Official NASA Dataset

**NASA C-MAPSS Jet Engine Simulated Data:**

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

**Dataset used:** `FD001`

FD001 contains:

* 100 training engine trajectories
* 100 test engine trajectories
* 1 operating condition
* 1 degradation/fault mode
* 3 operational settings
* 21 sensor measurements

Therefore, the raw observation data contains **26 columns**:

```text
1 Engine / Unit Number
1 Operating Cycle
3 Operational Settings
21 Sensor Measurements
----------------------
26 Raw Columns
```

The training data contains complete engine trajectories up to failure.

The test data contains partial engine trajectories together with a separate file containing the final RUL value for each test engine.

### Required FD001 files

AeroSense uses:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

---

# 3. Why FD001?

NASA's C-MAPSS dataset contains multiple subsets:

```text
FD001
FD002
FD003
FD004
```

AeroSense currently focuses on **FD001**.

FD001 provides a well-defined starting point for learning the complete RUL-prediction workflow because it contains:

* a single operating condition;
* a single degradation mode;
* multiple engine trajectories;
* complete run-to-failure training trajectories.

The project can later be extended to FD002, FD003, and FD004.

The current implementation intentionally keeps the scope focused on FD001 so that the complete machine-learning pipeline can be understood and demonstrated clearly.

---

# 4. Data Availability and GitHub

The raw NASA dataset is **not committed to this GitHub repository**.

The repository contains the source code required to process the dataset.

The project expects the dataset files inside:

```text
data/raw/
```

with:

```text
data/raw/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

The data pipeline checks whether these files are available locally.

If they are missing, the pipeline attempts to obtain the C-MAPSS archive automatically. If automatic retrieval is unavailable or the archive structure differs, the FD001 files can be downloaded manually from the official NASA dataset page and placed inside:

```text
data/raw/
```

### Why is the dataset not stored in GitHub?

Keeping the raw dataset outside the repository keeps the Git repository lightweight and avoids redistributing the complete dataset unnecessarily.

The `.gitignore` file excludes:

```text
data/raw/*
artifacts/*
reports/*
```

while allowing these directories to exist locally during execution.

---

# 5. Understanding the RUL Target

## Training Data

For each training engine:

```text
RUL = Maximum cycle of that engine - Current cycle
```

For example:

```text
Maximum cycle = 200

Current cycle    RUL
-------------   ----
20              180
50              150
100             100
150              50
190              10
200               0
```

AeroSense caps the training target at:

```text
MAX_RUL = 125
```

Therefore, very large early-life RUL values are clipped to 125.

This prevents extremely large early-life target values from dominating the learning problem.

---

## Test Data

NASA provides one final RUL value for each test engine.

For every observation in a test trajectory, AeroSense reconstructs the corresponding RUL using:

```text
RUL =
Final observed test cycle
- Current cycle
+ NASA final RUL
```

For the final benchmark-style test evaluation, AeroSense uses the **last observed row of each test engine**.

This corresponds to the latest available observation for each test engine.

---

# 6. Machine Learning Approach

AeroSense uses:

```text
Random Forest Regressor
```

Random Forest is an ensemble machine-learning algorithm composed of multiple decision trees.

A decision tree learns relationships by repeatedly splitting the data according to feature values.

Conceptually:

```text
if sensor_3 < threshold
       |
   ┌───┴───┐
   ↓       ↓
 left     right
```

Many decision trees are trained using different subsets of the data/features.

For regression, the predictions from the individual trees are combined to produce the final prediction.

Conceptually:

```text
Tree 1 → prediction
Tree 2 → prediction
Tree 3 → prediction
...
Tree 100 → prediction
             ↓
       Combined prediction
             ↓
          RUL value
```

---

# 7. Why Random Forest?

Random Forest was selected because it is a strong and understandable baseline for numerical tabular data.

Important characteristics relevant to AeroSense include:

* Captures non-linear relationships.
* Can model interactions between sensor measurements.
* Requires relatively little preprocessing.
* Does not require feature scaling.
* Provides feature-importance values.
* Works naturally as a regression model.
* Is relatively easy to explain in a technical interview.
* Provides a useful balance between model capability and interpretability for a first-year project.

Unlike distance-based or gradient-based methods that may be sensitive to feature scales, tree-based models can generally work directly with the numerical sensor values without normalizing every feature.

---

# 8. Data Processing Pipeline

The complete AeroSense pipeline is:

```text
NASA C-MAPSS FD001
        │
        ▼
Load raw files
        │
        ▼
Parse telemetry data
        │
        ▼
Calculate training RUL
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
Train Random Forest
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
1 Engine ID
1 Operating Cycle
3 Operational Settings
21 Sensor Measurements
```

The model does **not** use the engine ID as a predictive feature.

Engine ID is an identifier:

```text
Engine 1
Engine 2
Engine 3
...
```

It does not represent a physical measurement of the engine.

Therefore, the model excludes:

```text
unit_number
RUL
```

from the model inputs.

The remaining telemetry-related numerical features are used as model inputs.

Features whose variance is effectively zero are also removed automatically.

This is determined from the training data rather than relying on a permanently hard-coded list.

---

# 10. Data Leakage Prevention

One important issue in datasets containing multiple observations from the same entity is **data leakage**.

A random row-level split could produce:

```text
Engine 17, cycle 20 → training
Engine 17, cycle 21 → validation
Engine 17, cycle 22 → training
Engine 17, cycle 23 → validation
```

The model would then see parts of the same engine trajectory in both training and validation.

This could make validation performance misleadingly optimistic.

AeroSense instead performs the split at the **engine level** using:

```text
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

This provides a more meaningful validation procedure for the problem.

---

# 11. Random Forest Configuration

The current baseline configuration is:

```text
n_estimators      = 100
max_depth         = 15
min_samples_split = 5
random_state      = 42
n_jobs            = -1
oob_score         = True
```

### Parameter Meaning

**`n_estimators = 100`**

Creates 100 decision trees in the forest.

**`max_depth = 15`**

Limits the maximum depth of each decision tree.

**`min_samples_split = 5`**

A node must contain at least five samples before it can be split.

**`random_state = 42`**

Makes the experiment reproducible.

**`n_jobs = -1`**

Allows scikit-learn to use all available CPU cores.

**`oob_score = True`**

Enables out-of-bag scoring during Random Forest training.

---

# 12. Evaluation Metrics

AeroSense reports four evaluation metrics:

1. RMSE
2. MAE
3. R²
4. NASA asymmetric score

---

## RMSE

**Root Mean Squared Error**

RMSE gives greater importance to larger prediction errors.

Its unit is:

```text
cycles
```

A lower RMSE indicates smaller prediction error.

Conceptually:

```text
RMSE = square root of the average squared prediction error
```

---

## MAE

**Mean Absolute Error**

MAE represents the average absolute difference between predicted RUL and actual RUL.

For example:

```text
Actual RUL     = 50
Predicted RUL  = 43

Absolute error = 7 cycles
```

MAE is easy to interpret because it is expressed directly in cycles.

---

## R²

**Coefficient of Determination**

R² measures how well the model explains variation in the target relative to a baseline based on the target mean.

A value closer to 1 generally indicates stronger explanatory performance on the evaluated data.

---

## NASA / PHM Asymmetric Score

RUL prediction does not treat every type of prediction error equally.

AeroSense therefore also calculates the asymmetric score used for C-MAPSS / PHM-style evaluation.

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

The asymmetric structure means that over-prediction and under-prediction receive different penalties.

---

# 13. Current Test Result

The current trained FD001 model produced the following **final-cycle test evaluation**:

| Metric     |            Result |
| ---------- | ----------------: |
| RMSE       | **17.660 cycles** |
| MAE        | **13.161 cycles** |
| R²         |         **0.819** |
| NASA Score |       **498.498** |

These results are from the current project run.

They are provided as reproducible project results and should not be interpreted as certified aviation-maintenance performance.

---

# 14. Feature Importance

Random Forest provides feature-importance values based on the trained ensemble.

AeroSense saves these values and uses them to identify influential telemetry features.

The model identifies the **five most important sensor features** for the interactive dashboard.

This allows the project to investigate:

> **Which sensor measurements contributed most to the model's predictions?**

The feature-importance report is generated locally as:

```text
reports/feature_importance.csv
```

The dashboard also displays the top model features visually.

---

# 15. Streamlit Dashboard

AeroSense includes an interactive **Streamlit** dashboard.

Launch it with:

```bash
python -m streamlit run app/streamlit_app.py
```

On Windows, the Python launcher can be used:

```bash
py -m streamlit run app/streamlit_app.py
```

The dashboard provides several interactive components.

---

## Engine Selection

The user can select a test engine by ID.

The dashboard loads the selected engine's latest available telemetry.

---

## Latest Telemetry

The dashboard starts with the engine's latest observed sensor values.

This provides the baseline input for the RUL prediction.

---

## Sensor Controls

The five most important sensor features can be modified using interactive sliders.

Changing these values allows the user to explore how changes in telemetry affect the model's prediction.

---

## Live RUL Prediction

The Random Forest model predicts RUL using the selected telemetry.

The dashboard displays the resulting predicted RUL in operating cycles.

---

## Baseline Comparison

The dashboard compares the current slider-based prediction with the selected engine's baseline prediction.

This helps demonstrate how changing important sensor values can affect the model output.

---

## RUL Gauge

The dashboard visually displays:

```text
Predicted Remaining Useful Life
```

in operating cycles.

---

## Feature Importance

The dashboard displays the most influential model features.

This provides a simple way to inspect which sensor measurements the trained Random Forest considered important.

---

## Telemetry Visualization

The dashboard provides a visual telemetry snapshot for the selected engine.

It also displays sensor values together with information such as:

* sensor identifier;
* sensor symbol;
* unit;
* physical meaning.

---

## Raw Telemetry Snapshot

The dashboard provides an expandable view containing the selected engine's telemetry information.

This makes it possible to inspect the underlying input values used by the application.

---

# 16. Dashboard Status Categories

For visualization purposes, the dashboard uses three educational status categories:

| Predicted RUL | Dashboard Status    |
| ------------: | ------------------- |
|   > 60 cycles | SAFE                |
|  21–60 cycles | MAINTENANCE WARNING |
|   ≤ 20 cycles | CRITICAL            |

These thresholds are **demo categories created for the project interface**.

They are not certified aviation maintenance limits and should not be interpreted as operational recommendations.

---

# 17. Project Structure

The current GitHub repository contains:

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

### Generated Locally

The following files are generated or populated during execution and are intentionally not committed to GitHub:

```text
data/
└── raw/
    ├── train_FD001.txt
    ├── test_FD001.txt
    ├── RUL_FD001.txt
    └── CMAPSSData.zip

artifacts/
├── model.joblib
└── metadata.joblib

reports/
├── metrics.json
├── feature_importance.csv
└── test_predictions.csv
```

These files can be reproduced locally by running the project.

---

# 18. What Each Source File Does

## `src/data_pipeline.py`

Responsible for the data-processing layer.

It:

* checks for the required NASA C-MAPSS files;
* attempts dataset retrieval when the files are missing;
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
* compares baseline and modified predictions;
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

Contains important design decisions made during the project, including:

* why feature scaling is not used;
* why engine-wise splitting is used;
* why raw data is not committed;
* how final-cycle testing is performed;
* why dashboard thresholds are educational only.

---

# 19. Installation

## Prerequisites

For the dependency versions specified in `requirements.txt`, use a compatible Python environment in the **3.10–3.13 range**.

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

```bash
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

The project expects the FD001 dataset files in:

```text
data/raw/
```

The required files are:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

The official dataset can be obtained from:

**NASA C-MAPSS Jet Engine Simulated Data**

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

If the files are already present locally, no additional dataset setup is required.

If they are missing, the training pipeline attempts automatic dataset retrieval. If that is unsuccessful, download the dataset manually from NASA and place the required FD001 files inside:

```text
data/raw/
```

---

# 21. Train the Model

From the repository root:

```bash
python -m src.model_train
```

On Windows:

```bash
py -m src.model_train
```

The training process is:

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

The dashboard requires the model artifacts produced by training.

Run:

```bash
python -m streamlit run app/streamlit_app.py
```

On Windows:

```bash
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

```bash
py -m pytest
```

The test suite currently covers the main data-processing and evaluation utilities.

The current local test result is:

```text
3 passed
```

---

# 24. Reproducibility

Several settings are fixed to make experiments reproducible.

The Random Forest uses:

```text
random_state = 42
```

The model configuration is explicitly defined in:

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

This allows the dashboard to use the same feature configuration as the trained model.

---

# 25. Important Project Limitations

AeroSense is an **educational machine-learning project**, not an aircraft-engine maintenance system.

Important limitations include:

### Simulated Data

C-MAPSS is a simulated engine degradation dataset rather than live aircraft telemetry.

### Single Dataset Subset

Only FD001 is currently implemented.

### Single Baseline Model

The current project focuses on Random Forest Regression rather than comparing multiple machine-learning algorithms.

### Demo Status Thresholds

The Safe / Maintenance Warning / Critical categories are interface demonstrations only.

### No Operational Deployment

The model should not be used to make real aviation maintenance decisions.

### Limited Generalization

Performance on the C-MAPSS dataset does not establish performance on real aircraft engines or other datasets.

---

# 26. Possible Future Extensions

The current project can be expanded in several directions.

### Dataset Extensions

```text
FD001
  ↓
FD002 / FD003 / FD004
```

### Possible Machine-Learning Extensions

* XGBoost;
* Gradient Boosting;
* Extra Trees;
* hyperparameter tuning;
* model comparison;
* ensemble approaches.

### Feature Engineering

* sliding-window features;
* sensor trends;
* rolling averages;
* degradation rates;
* time-series features.

### Explainability

* SHAP;
* more detailed feature analysis;
* individual prediction explanations.

### Advanced Modeling

* uncertainty estimation;
* sequence-based models;
* recurrent neural networks;
* temporal deep-learning models.

### Engineering Extensions

* experiment tracking;
* continuous integration;
* automated testing;
* dashboard deployment;
* support for multiple C-MAPSS operating conditions.

These are possible future extensions rather than requirements of the current implementation.

---

# 27. Reproducing the Current Result

A complete local setup is:

```bash
git clone https://github.com/sushanth0903-CODE/aerosense-rul.git
cd aerosense-rul

python -m venv .venv
```

Activate the environment.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Make sure the FD001 dataset files are available in:

```text
data/raw/
```

Train the model:

```bash
python -m src.model_train
```

Run the dashboard:

```bash
python -m streamlit run app/streamlit_app.py
```

### Windows Python Launcher

```bash
git clone https://github.com/sushanth0903-CODE/aerosense-rul.git
cd aerosense-rul

py -m venv .venv
.venv\Scripts\activate

py -m pip install -r requirements.txt

py -m src.model_train

py -m streamlit run app/streamlit_app.py
```

---

# 28. References

1. **NASA Open Data — CMAPSS Jet Engine Simulated Data**

   https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

2. Saxena, A., Goebel, K., Simon, D., & Eklund, N.
   **Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation**, PHM08.

3. **Scikit-learn — RandomForestRegressor**

   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

4. **Streamlit Documentation**

   https://docs.streamlit.io/

---

# Project Summary

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

## Final Result

AeroSense demonstrates a complete end-to-end machine-learning workflow:

**raw turbofan telemetry → RUL target construction → leakage-aware validation → Random Forest regression → evaluation → model artifacts → feature analysis → interactive RUL dashboard**

The current FD001 final-cycle evaluation produced:

```text
RMSE       = 17.660 cycles
MAE        = 13.161 cycles
R²         = 0.819
NASA Score = 498.498
```

AeroSense is designed as an educational demonstration of how a machine-learning model can transform sensor telemetry into an interpretable Remaining Useful Life prediction.
