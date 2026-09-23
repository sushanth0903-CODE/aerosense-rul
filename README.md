# ✈️ AeroSense

## NASA C-MAPSS Turbofan Remaining Useful Life Prediction using Random Forest

> **AeroSense predicts how many operating cycles a simulated turbofan engine may have remaining before failure using sensor telemetry and a Random Forest regression model.**

AeroSense is an end-to-end **predictive-maintenance machine-learning project** built using the **NASA C-MAPSS FD001 turbofan engine dataset**.

The project demonstrates the complete workflow of a practical machine-learning system:

```text
Raw Engine Telemetry
        ↓
Data Preparation
        ↓
RUL Target Construction
        ↓
Leakage-Aware Train/Validation Split
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Model + Metadata + Reports
        ↓
Interactive Streamlit Dashboard
```

The goal of AeroSense is not simply to train a machine-learning model, but to demonstrate **how raw sensor data can be transformed into a meaningful Remaining Useful Life (RUL) prediction and presented through an interactive application**.

---

# 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem](#2-the-problem)
3. [What is Remaining Useful Life?](#3-what-is-remaining-useful-life)
4. [Why RUL Prediction is a Regression Problem](#4-why-rul-prediction-is-a-regression-problem)
5. [Dataset](#5-dataset)
6. [Why NASA C-MAPSS FD001?](#6-why-nasa-c-mapps-fd001)
7. [What the Dataset Looks Like](#7-what-the-dataset-looks-like)
8. [How AeroSense Creates the RUL Target](#8-how-aerosense-creates-the-rul-target)
9. [Complete Machine-Learning Pipeline](#9-complete-machine-learning-pipeline)
10. [Feature Preparation](#10-feature-preparation)
11. [Preventing Data Leakage](#11-preventing-data-leakage)
12. [Why Random Forest?](#12-why-random-forest)
13. [How Random Forest Works](#13-how-random-forest-works)
14. [Random Forest Configuration](#14-random-forest-configuration)
15. [Why Feature Scaling is Not Used](#15-why-feature-scaling-is-not-used)
16. [Model Evaluation](#16-model-evaluation)
17. [Current Results](#17-current-results)
18. [Feature Importance](#18-feature-importance)
19. [Streamlit Dashboard](#19-streamlit-dashboard)
20. [Dashboard Status Categories](#20-dashboard-status-categories)
21. [Project Architecture](#21-project-architecture)
22. [Repository Structure](#22-repository-structure)
23. [What Each File Does](#23-what-each-file-does)
24. [Installation](#24-installation)
25. [Automatic Dataset Setup](#25-automatic-dataset-setup)
26. [Train the Model](#26-train-the-model)
27. [Run the Dashboard](#27-run-the-dashboard)
28. [Run the Tests](#28-run-the-tests)
29. [Generated Files](#29-generated-files)
30. [Reproducibility](#30-reproducibility)
31. [Technical Interview Guide](#31-technical-interview-guide)
32. [Limitations](#32-limitations)
33. [Future Improvements](#33-future-improvements)
34. [References](#34-references)
35. [Project Summary](#35-project-summary)

---

# 1. Project Overview

## 🎯 Project Goal

The objective of AeroSense is:

> **Given the current telemetry of a turbofan engine, estimate how many operating cycles remain before the engine reaches the end of its simulated useful life.**

This is a classic **predictive maintenance** problem.

Instead of waiting until a machine fails:

```text
Failure happens
      ↓
Maintenance
```

predictive maintenance tries to estimate degradation in advance:

```text
Sensor measurements
        ↓
Machine-learning model
        ↓
Estimated remaining life
        ↓
Maintenance planning
```

AeroSense demonstrates this concept using simulated turbofan-engine telemetry.

---

# 2. The Problem

Imagine an aircraft engine being monitored continuously.

During operation, sensors measure quantities such as:

* temperatures;
* pressures;
* fan speed;
* core speed;
* fuel-related quantities;
* coolant flow;
* other engine parameters.

As an engine gradually degrades, some of these measurements can change.

The challenge is:

> **Can we learn the relationship between those measurements and how close the engine is to failure?**

AeroSense attempts to answer that question with machine learning.

---

# 3. What is Remaining Useful Life?

**Remaining Useful Life**, usually abbreviated as **RUL**, means:

> The estimated amount of operating life remaining before an engine reaches its failure point.

In C-MAPSS, engine operation is measured in **cycles**.

For example:

```text
Current cycle = 140
Failure cycle = 200
```

Therefore:

```text
RUL = Failure cycle - Current cycle

RUL = 200 - 140

RUL = 60 cycles
```

So the engine has:

```text
60 operating cycles remaining
```

before the simulated failure point.

---

# 4. Why RUL Prediction is a Regression Problem

There are two broad kinds of machine-learning prediction:

### Classification

Classification predicts a category.

Example:

```text
Healthy
Warning
Critical
```

### Regression

Regression predicts a numerical value.

Example:

```text
Predicted RUL = 43.7 cycles
```

AeroSense predicts a numerical quantity:

```text
Remaining Useful Life
```

Therefore:

> **AeroSense is a regression problem.**

The Random Forest model used in this project is specifically:

```text
RandomForestRegressor
```

---

# 5. Dataset

AeroSense uses the:

## NASA C-MAPSS Jet Engine Simulated Data

Official NASA dataset page:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

The project currently uses:

```text
FD001
```

C-MAPSS contains several commonly used subsets:

```text
FD001
FD002
FD003
FD004
```

AeroSense currently focuses on **FD001**.

---

# 6. Why NASA C-MAPSS FD001?

FD001 is a useful starting point for understanding the complete RUL-prediction workflow because it provides:

* multiple simulated engine trajectories;
* complete run-to-failure training trajectories;
* a single operating condition;
* a single degradation/fault mode;
* multiple sensor measurements over time.

The simplified structure makes it easier to focus on the machine-learning concepts before introducing additional operating-condition complexity.

The current project deliberately keeps the implementation focused on FD001.

Future versions could extend the same pipeline to:

```text
FD002
FD003
FD004
```

---

# 7. What the Dataset Looks Like

FD001 contains:

```text
100 training engines
100 test engines
1 operating condition
1 degradation/fault mode
3 operational settings
21 sensor measurements
```

Each raw observation contains:

```text
1 engine identifier
1 operating cycle
3 operational settings
21 sensors
```

Therefore:

```text
1 + 1 + 3 + 21 = 26 raw columns
```

Conceptually, a row looks like:

```text
Engine  | Cycle | Settings | Sensor 1 | Sensor 2 | ... | Sensor 21
--------------------------------------------------------------------
1       | 1     | ...      | ...      | ...      | ... | ...
1       | 2     | ...      | ...      | ...      | ... | ...
1       | 3     | ...      | ...      | ...      | ... | ...
...
```

The engine is observed over many cycles.

---

# 8. How AeroSense Creates the RUL Target

The original training dataset contains telemetry, but the model also needs a **target** to learn.

That target is:

```text
RUL
```

## Training Data

For each training engine, AeroSense calculates:

```text
RUL = Maximum cycle of that engine - Current cycle
```

Example:

```text
Engine maximum cycle = 200

Current cycle     RUL
-------------    ----
20               180
50               150
100              100
150               50
190               10
200                0
```

At the final cycle:

```text
RUL = 0
```

because the engine has reached the end of its simulated life.

---

## RUL Capping

AeroSense caps the training target at:

```text
MAX_RUL = 125
```

Therefore:

```text
RUL = 180 → 125
RUL = 150 → 125
RUL = 130 → 125
RUL = 100 → 100
RUL = 60  → 60
```

This prevents very large early-life RUL values from dominating the regression problem.

The cap is applied only to the training target construction.

---

## Test Data

The test engines do not contain complete trajectories.

Instead, NASA provides:

```text
test_FD001.txt
RUL_FD001.txt
```

`RUL_FD001.txt` gives the remaining useful life **after the final observed test cycle** for each test engine.

AeroSense reconstructs row-level test RUL using:

```text
RUL =
Final observed test cycle
- Current cycle
+ NASA final RUL
```

The final benchmark-style test evaluation then uses:

> **The last observed row of each test engine.**

This gives one prediction per test engine, matching the intended final-cycle evaluation setup.

---

# 9. Complete Machine-Learning Pipeline

The complete AeroSense pipeline is:

```text
NASA C-MAPSS FD001
        │
        ▼
Automatic dataset retrieval
        │
        ▼
Read raw telemetry
        │
        ▼
Assign column names
        │
        ▼
Create training RUL
        │
        ▼
Reconstruct test RUL
        │
        ▼
Split training engines
        │
        ▼
Select model features
        │
        ▼
Remove near-zero variance features
        │
        ▼
Train Random Forest
        │
        ├───────────────► Feature Importance
        │
        ▼
Validation predictions
        │
        ▼
Validation metrics
        │
        ▼
Final-cycle test predictions
        │
        ▼
Test metrics
        │
        ▼
Save model + metadata + reports
        │
        ▼
Streamlit Dashboard
```

In simple terms:

> **Data → Target → Features → Model → Prediction → Evaluation → Dashboard**

---

# 10. Feature Preparation

The raw data contains:

```text
unit_number
cycle
operational_setting_1
operational_setting_2
operational_setting_3
sensor_1
sensor_2
...
sensor_21
```

The model does **not** use:

```text
unit_number
```

as a prediction feature.

Why?

Because `unit_number` is only an identifier:

```text
Engine 1
Engine 2
Engine 3
...
```

It does not represent a physical measurement of the engine.

The target:

```text
RUL
```

is also not used as an input because it is the value the model is trying to predict.

Therefore:

```text
Excluded:
    unit_number
    RUL

Used:
    cycle
    operational settings
    informative sensor measurements
```

---

## Near-Zero Variance Features

A feature with almost no variation provides little useful information for a model.

For example, imagine:

```text
Feature A:

10
10
10
10
10
10
```

The feature never changes.

AeroSense calculates feature variance on the training data and removes features whose variance is effectively zero:

```text
variance > 1e-12
```

This avoids feeding useless constant-like features to the model.

---

# 11. Preventing Data Leakage

One of the most important technical decisions in AeroSense is preventing **data leakage**.

## What is data leakage?

Data leakage happens when information from the validation/test set unintentionally influences model training.

For this dataset, a particularly dangerous mistake would be to randomly split individual rows.

For example:

```text
Training:

Engine 17, cycle 20
Engine 17, cycle 21
Engine 17, cycle 24

Validation:

Engine 17, cycle 22
Engine 17, cycle 23
```

Now the model has already seen the same engine's trajectory during training.

That can make the validation result look better than it really is.

---

## AeroSense Solution

AeroSense splits the data by:

```text
ENGINE
```

rather than by random rows.

The project uses:

```text
GroupShuffleSplit
```

with:

```text
20% validation
random_state = 42
```

Therefore:

```text
Engine 1 → Training
Engine 2 → Training
Engine 3 → Validation
Engine 4 → Training
...
```

A complete engine belongs to only one side of the split.

This produces a more meaningful validation experiment for the problem.

---

# 12. Why Random Forest?

AeroSense uses:

```text
Random Forest Regressor
```

Random Forest was selected because it is well suited to the project's numerical/tabular telemetry data while remaining relatively straightforward to understand and explain.

Useful characteristics include:

### 1. Non-linear relationships

Engine degradation may not behave like a simple straight-line equation.

Random Forest can learn non-linear relationships.

### 2. Feature interactions

A sensor may not be informative by itself but can become informative when combined with another sensor.

Tree-based methods can model these interactions.

### 3. No feature scaling requirement

Tree-based models do not require inputs to be normalized in the same way as many distance-based or gradient-based algorithms.

### 4. Good interpretability for a student project

Decision trees are easier to explain than many black-box models.

### 5. Feature importance

Random Forest provides feature-importance values that can be visualized in the dashboard.

### 6. Reasonable baseline

Random Forest provides a practical baseline before moving toward more advanced models such as gradient boosting or deep-learning sequence models.

---

# 13. How Random Forest Works

A single decision tree makes predictions through a sequence of learned rules.

Conceptually:

```text
Is sensor_3 < threshold?
          │
      ┌───┴───┐
      │       │
     Yes      No
      │       │
      ▼       ▼
   Branch A  Branch B
```

A Random Forest creates many trees.

For example:

```text
Tree 1 → 42 cycles
Tree 2 → 48 cycles
Tree 3 → 45 cycles
Tree 4 → 41 cycles
...
Tree 100 → 46 cycles
```

For regression, the forest combines the tree predictions to produce a final prediction.

Conceptually:

```text
100 decision trees
        ↓
Combine predictions
        ↓
Final RUL prediction
```

So AeroSense does not depend on a single decision tree.

It uses an **ensemble of trees**.

---

# 14. Random Forest Configuration

The current AeroSense configuration is:

```text
n_estimators      = 100
max_depth         = 15
min_samples_split = 5
random_state      = 42
n_jobs            = -1
oob_score         = True
```

## Parameter explanation

### `n_estimators = 100`

The forest contains:

```text
100 decision trees
```

### `max_depth = 15`

Each tree can grow to a maximum depth of:

```text
15
```

This limits the complexity of individual trees.

### `min_samples_split = 5`

A node must contain at least:

```text
5 samples
```

before it can be split.

### `random_state = 42`

This fixes the random seed and improves reproducibility.

### `n_jobs = -1`

The model can use all available CPU cores for training.

### `oob_score = True`

Random Forest keeps track of out-of-bag samples during training and computes an OOB estimate internally.

---

# 15. Why Feature Scaling is Not Used

Many machine-learning algorithms benefit from feature scaling.

For example:

```text
sensor A: 0.1 → 1.2
sensor B: 1000 → 5000
```

For distance-based models, these very different numeric ranges can matter.

Random Forest works differently.

Decision trees primarily make decisions such as:

```text
Is sensor_7 < 42.5?
```

The split is based on ordering and thresholds rather than Euclidean distance.

Therefore, explicit normalization or standardization is **not required** for this Random Forest pipeline.

This keeps the implementation simpler and preserves the original sensor-value meaning.

---

# 16. Model Evaluation

AeroSense evaluates the model using four metrics:

```text
RMSE
MAE
R²
NASA Asymmetric Score
```

Each metric answers a slightly different question.

---

## 16.1 RMSE

### Root Mean Squared Error

RMSE penalizes larger errors more strongly.

Conceptually:

```text
Prediction error
        ↓
Square error
        ↓
Average
        ↓
Square root
        ↓
RMSE
```

Its unit is:

```text
cycles
```

Lower RMSE indicates smaller prediction error.

---

## 16.2 MAE

### Mean Absolute Error

MAE measures the average absolute prediction error.

Example:

```text
Actual RUL    = 50
Predicted RUL = 43

Error = 7 cycles
```

MAE is therefore easy to interpret:

> On average, how many cycles away are the predictions from the true values?

Lower MAE is better.

---

## 16.3 R²

### Coefficient of Determination

R² measures how well the model explains variation in the target relative to a mean-based baseline.

In general:

```text
Closer to 1
    ↓
Stronger explanatory performance
```

An R² value should always be interpreted in the context of the dataset and evaluation procedure.

---

## 16.4 NASA / PHM Asymmetric Score

RUL prediction has an important characteristic:

> Under-predicting and over-predicting are not necessarily treated as equally costly.

AeroSense therefore calculates the asymmetric scoring function used in the C-MAPSS / PHM evaluation context.

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

The final score is the sum over all evaluated engines.

Important interpretation:

```text
Lower score → better
Perfect predictions → 0
```

The score penalizes the two directions of error differently.

---

# 17. Current Results

The current AeroSense model was evaluated using the final observed cycle of each FD001 test engine.

### Final-cycle test evaluation

| Metric         |            Result |
| -------------- | ----------------: |
| **RMSE**       | **17.660 cycles** |
| **MAE**        | **13.161 cycles** |
| **R²**         |         **0.819** |
| **NASA Score** |       **498.498** |

Validation results from the current run:

| Metric                    |            Result |
| ------------------------- | ----------------: |
| **Validation RMSE**       | **15.999 cycles** |
| **Validation MAE**        | **10.840 cycles** |
| **Validation R²**         |         **0.853** |
| **Validation NASA Score** |     **22627.066** |

These values are **reproducible project results for the current configuration and dataset**, not guarantees of performance on real aircraft engines.

---

# 18. Feature Importance

One benefit of Random Forest is that it provides feature-importance information.

AeroSense calculates importance values for the model features and saves them to:

```text
reports/feature_importance.csv
```

The dashboard visualizes the:

```text
Top 10 features
```

This allows us to ask:

> Which telemetry variables did the trained Random Forest rely on most during learning?

The five most important **sensor features** are also identified and exposed through the interactive dashboard.

---

## Important Interpretation

Feature importance does **not** automatically mean:

```text
"This sensor causes engine degradation."
```

It means:

```text
"The trained model relied more heavily on this feature
relative to the other features in this model."
```

Correlation, causation, and model importance are different concepts.

---

# 19. Streamlit Dashboard

AeroSense includes an interactive Streamlit application.

The dashboard lets the user move from:

```text
Raw telemetry
      ↓
Model
      ↓
Prediction
```

through a visual interface.

The dashboard includes:

### ✈️ Engine Selection

Select one of the test engines by ID.

---

### 📡 Latest Telemetry

The dashboard takes the selected engine's:

```text
latest observed cycle
```

as the baseline input.

---

### 🎛️ Sensor Controls

The five most important sensor features can be modified using sliders.

This creates an educational "what-if" experiment:

```text
Original telemetry
        ↓
Change sensor value
        ↓
Run model again
        ↓
Observe changed RUL prediction
```

---

### 🔢 Predicted RUL

The dashboard displays the current model prediction in:

```text
cycles
```

---

### 📊 RUL Gauge

A visual gauge shows the predicted remaining useful life.

---

### 📈 Change from Baseline

The dashboard also calculates:

```text
Adjusted prediction
-
Original prediction
```

This shows how much the selected sensor modifications changed the model's output.

---

### 🧠 Feature Importance

The dashboard displays the top model features.

---

### 📡 Telemetry Visualization

The five adjustable sensor values are visualized using an interactive chart.

---

### 📋 Sensor Information Table

The dashboard shows:

* sensor identifier;
* engineering symbol;
* current value;
* unit;
* physical meaning.

---

### 📄 Raw Telemetry Snapshot

An expandable section shows the underlying selected engine telemetry.

---

### 🎓 Built-in Explanation

The dashboard also contains an educational explanation of:

* the RUL target;
* the Random Forest;
* the evaluation process;
* the sensor interaction demo.

This allows someone unfamiliar with the project to understand what the interface is doing.

---

# 20. Dashboard Status Categories

For educational visualization, AeroSense groups predicted RUL into:

|    Predicted RUL | Status                 |
| ---------------: | ---------------------- |
|  **> 60 cycles** | 🟢 SAFE                |
| **21–60 cycles** | 🟡 MAINTENANCE WARNING |
|  **≤ 20 cycles** | 🔴 CRITICAL            |

### ⚠️ Important

These are:

> **Demonstration categories created for the dashboard interface.**

They are **not certified aviation maintenance limits**.

They must not be interpreted as real aircraft maintenance recommendations.

---

# 21. Project Architecture

The repository follows a simple separation of responsibilities:

```text
                    ┌──────────────────────────┐
                    │ NASA C-MAPSS FD001 Data │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   data_pipeline.py       │
                    │                          │
                    │ • Download dataset       │
                    │ • Read files             │
                    │ • Create RUL              │
                    │ • Prepare test RUL       │
                    │ • Split by engine        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     model_train.py       │
                    │                          │
                    │ • Select features        │
                    │ • Train Random Forest    │
                    │ • Predict                 │
                    │ • Save model             │
                    └───────┬──────────┬───────┘
                            │          │
              ┌─────────────┘          └─────────────┐
              ▼                                      ▼
┌──────────────────────────┐            ┌──────────────────────────┐
│      evaluate.py         │            │       Reports            │
│                          │            │                          │
│ • RMSE                   │            │ • metrics.json            │
│ • MAE                    │            │ • feature_importance.csv │
│ • R²                     │            │ • test_predictions.csv   │
│ • NASA score             │            └──────────────────────────┘
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│              Streamlit Dashboard                   │
│                                                    │
│ Engine selection → Telemetry → Prediction →        │
│ Gauge → Feature importance → Sensor interaction   │
└────────────────────────────────────────────────────┘
```

---

# 22. Repository Structure

The GitHub repository is organized as:

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

Generated locally:

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

The generated dataset, model files, and reports are intentionally excluded from version control.

---

# 23. What Each File Does

## `src/data_pipeline.py`

This is the **data layer**.

It is responsible for:

* checking whether FD001 exists;
* automatically downloading the official archive if required;
* handling the nested archive structure;
* extracting the three FD001 files;
* reading the raw telemetry;
* assigning column names;
* calculating training RUL;
* reconstructing test RUL;
* splitting engines for validation;
* identifying informative features.

---

## `src/model_train.py`

This is the **training layer**.

It:

1. loads FD001;
2. creates the engine-wise training/validation split;
3. selects model features;
4. removes near-zero variance features;
5. creates the Random Forest;
6. trains the model;
7. predicts validation RUL;
8. evaluates final-cycle test RUL;
9. calculates feature importance;
10. saves the trained model;
11. saves metadata;
12. saves reports.

---

## `src/evaluate.py`

This is the **evaluation layer**.

It provides:

* RMSE;
* MAE;
* R²;
* asymmetric NASA score;
* combined metric calculation;
* metric saving;
* dashboard health-status logic.

---

## `app/streamlit_app.py`

This is the **application layer**.

It:

* loads the trained model;
* loads model metadata;
* loads FD001 telemetry;
* allows engine selection;
* provides sensor sliders;
* generates predictions;
* compares baseline and adjusted predictions;
* displays the RUL gauge;
* displays feature importance;
* displays sensor information;
* displays telemetry snapshots.

---

## `tests/test_data_pipeline.py`

Tests important data-processing behavior, including RUL-related functionality.

---

## `tests/test_evaluate.py`

Tests the evaluation utilities, including regression metrics and the asymmetric NASA scoring logic.

---

## `PROJECT_NOTES.md`

Contains project design decisions and technical notes, including:

* why Random Forest is used;
* why scaling is not required;
* why engine-wise splitting is used;
* why raw data is not committed;
* how final-cycle evaluation works;
* why dashboard thresholds are educational only.

---

# 24. Installation

## Requirements

AeroSense currently uses:

```text
Python
NumPy
Pandas
Scikit-learn
Joblib
Plotly
Streamlit
Pytest
```

The repository pins exact dependency versions in:

```text
requirements.txt
```

### Python

Use:

```text
Python 3.11+
```

The current repository has been successfully tested on:

```text
Python 3.14
```

---

## Option A — Clone with Git

```bash
git clone https://github.com/sushanth0903-CODE/aerosense-rul.git
cd aerosense-rul
```

---

## Option B — Download ZIP from GitHub

You can also use:

```text
GitHub
   ↓
Code
   ↓
Download ZIP
   ↓
Extract
   ↓
Open terminal in the extracted folder
```

No pre-existing NASA dataset is required.

---

## Create a virtual environment

### Windows

```bash
py -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install dependencies

### Windows

```bash
py -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m pip install -r requirements.txt
```

---

# 25. Automatic Dataset Setup

One of the important features of AeroSense is that the project does **not require the user to manually download the NASA dataset before the first run**.

When the training pipeline starts, it checks:

```text
data/raw/
```

for:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

If they are not present, AeroSense automatically:

```text
1. Downloads the official C-MAPSS archive
2. Opens the downloaded ZIP
3. Searches for the FD001 files
4. Handles the nested CMAPSSData.zip structure
5. Extracts the required files
6. Continues with model training
```

The user therefore normally only needs to run:

```bash
py -m src.model_train
```

on Windows.

### What a first run looks like

```text
NASA C-MAPSS archive not found. Downloading it now...
NASA C-MAPSS FD001 dataset is ready.

=== AeroSense training complete ===
...
```

After the first successful download, the dataset is reused locally.

---

## Manual fallback

Automatic retrieval requires internet access.

If a user's network blocks the dataset host, the FD001 files can still be downloaded from the official NASA dataset page:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Then place:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

inside:

```text
data/raw/
```

This is only a fallback; it is **not required in the normal setup**.

---

# 26. Train the Model

From the project root:

### Windows

```bash
py -m src.model_train
```

### macOS / Linux

```bash
python -m src.model_train
```

The command performs the full workflow:

```text
Check dataset
      ↓
Download if needed
      ↓
Load FD001
      ↓
Build RUL targets
      ↓
Split engines
      ↓
Select features
      ↓
Train Random Forest
      ↓
Evaluate validation set
      ↓
Evaluate final test cycles
      ↓
Save model
      ↓
Save metadata
      ↓
Save reports
```

---

# 27. Run the Dashboard

The dashboard requires the trained model artifacts.

Run:

### Windows

```bash
py -m streamlit run app/streamlit_app.py
```

### macOS / Linux

```bash
python -m streamlit run app/streamlit_app.py
```

Streamlit will provide a local address similar to:

```text
http://localhost:8501
```

Open that address in a browser.

---

# 28. Run the Tests

AeroSense includes automated tests.

Run:

### Windows

```bash
py -m pytest
```

### macOS / Linux

```bash
python -m pytest
```

The current test suite covers core data-processing and evaluation functionality.

Current local result:

```text
3 passed
```

---

# 29. Generated Files

Running the training pipeline creates:

## Model artifacts

```text
artifacts/model.joblib
artifacts/metadata.joblib
```

### `model.joblib`

Contains the trained:

```text
RandomForestRegressor
```

### `metadata.joblib`

Contains information needed by the dashboard, including:

* selected features;
* top sensor features;
* sensor descriptions;
* feature ranges;
* training RUL cap;
* Random Forest parameters.

---

## Reports

```text
reports/metrics.json
reports/feature_importance.csv
reports/test_predictions.csv
```

### `metrics.json`

Contains:

* validation metrics;
* final-cycle test metrics;
* engine counts;
* evaluation information.

### `feature_importance.csv`

Contains the Random Forest feature-importance values.

### `test_predictions.csv`

Contains:

* engine number;
* final observed cycle;
* actual RUL;
* predicted RUL;
* absolute error.

---

# 30. Reproducibility

AeroSense uses explicit configuration values to make the experiment reproducible.

The Random Forest uses:

```text
random_state = 42
```

The model configuration is stored in:

```text
src/model_train.py
```

The repository also records important model metadata in:

```text
artifacts/metadata.joblib
```

This helps ensure that the dashboard and trained model use the same feature definitions.

---

## Reproducible setup

A fresh setup can be reproduced with:

```bash
git clone https://github.com/sushanth0903-CODE/aerosense-rul.git
cd aerosense-rul

py -m venv .venv
.venv\Scripts\activate

py -m pip install -r requirements.txt

py -m src.model_train

py -m streamlit run app/streamlit_app.py
```

The NASA dataset is downloaded automatically on the first training run when it is not already present.

---

# 31. Technical Interview Guide

AeroSense is intentionally designed so that its important decisions can be explained technically.

A person presenting the project should be comfortable answering the following questions.

---

## What is the problem?

Predict the remaining number of operating cycles before simulated engine failure.

---

## Why is it regression?

Because the output is a continuous numerical value:

```text
RUL = 43.7 cycles
```

rather than a class label.

---

## What is the target?

```text
RUL
```

---

## How is training RUL calculated?

```text
Maximum engine cycle - Current cycle
```

with a training cap of:

```text
125 cycles
```

---

## Why remove engine ID?

Because engine ID identifies an engine but does not physically describe its state.

---

## Why split by engine instead of rows?

Because rows from the same engine are strongly related over time.

Row-level splitting can allow the same engine to appear in both training and validation.

Engine-level grouping reduces this leakage risk.

---

## Why use `GroupShuffleSplit`?

Because it allows the split to be performed using:

```text
unit_number
```

as the grouping variable.

---

## Why Random Forest?

Because it:

* handles non-linear relationships;
* can learn feature interactions;
* works well with numerical/tabular data;
* does not require feature scaling;
* provides feature importance;
* is easier to explain than many advanced models.

---

## Why no normalization?

Because Random Forest decision trees split using feature thresholds and do not rely on distance calculations between samples.

---

## Why cap RUL at 125?

To reduce the influence of very large early-life RUL values and focus the model on a more useful bounded target range.

---

## Why is the final-cycle test evaluation used?

Because the C-MAPSS test set provides a final RUL value for each test engine corresponding to the final observed test cycle.

Therefore, the project evaluates the latest observation of each test engine.

---

## Why use RMSE and MAE together?

Because they provide different views of error:

```text
MAE
→ average absolute error

RMSE
→ penalizes larger errors more heavily
```

---

## Why use the NASA asymmetric score?

Because the evaluation context distinguishes the penalties for under-prediction and over-prediction.

---

## What does R² = 0.819 mean?

It indicates that the model explains substantial variation in the evaluated test target relative to a mean-based baseline.

It does **not** mean:

```text
"The model is 81.9% accurate."
```

R² is not an accuracy percentage.

---

## What does feature importance mean?

It indicates how much the trained Random Forest relied on each feature during its learned tree splits.

It does not automatically prove causation.

---

## Why is this not a real aircraft maintenance system?

Because:

```text
NASA C-MAPSS
```

contains simulated engine degradation rather than live operational aircraft data.

A real deployment would require much more:

* real-world validation;
* safety analysis;
* uncertainty estimation;
* domain expertise;
* monitoring;
* regulatory compliance;
* testing under real operating conditions.

---

# 32. Limitations

AeroSense is an educational machine-learning project.

Important limitations include:

## 1. Simulated dataset

C-MAPSS is a simulated engine-degradation dataset.

Real aircraft telemetry can contain noise, missing values, changing operating conditions, sensor faults, maintenance events, and many other complications.

---

## 2. Only FD001 is currently supported

The project currently focuses on:

```text
FD001
```

rather than the complete C-MAPSS collection.

---

## 3. One baseline model

The current implementation uses:

```text
Random Forest
```

A more complete study would compare several approaches.

---

## 4. No uncertainty estimation

The current dashboard produces a point prediction:

```text
43.7 cycles
```

It does not provide a statistically calibrated prediction interval such as:

```text
43.7 ± 8.2 cycles
```

---

## 5. Dashboard status thresholds are educational

The categories:

```text
SAFE
MAINTENANCE WARNING
CRITICAL
```

are demonstration labels.

They are not real aviation maintenance rules.

---

## 6. Generalization is not guaranteed

Good performance on C-MAPSS does not guarantee performance on:

```text
real aircraft engines
different engine families
different sensor systems
different operating conditions
```

---

# 33. Future Improvements

AeroSense can be expanded substantially.

## Dataset expansion

Support:

```text
FD001
FD002
FD003
FD004
```

This would introduce more varied operating conditions and degradation behavior.

---

## More machine-learning models

Potential comparisons include:

```text
Random Forest
Extra Trees
Gradient Boosting
XGBoost
LightGBM
```

---

## Hyperparameter optimization

Instead of using a fixed baseline configuration, the project could use:

```text
Grid Search
Random Search
Bayesian optimization
```

to investigate improved model configurations.

---

## Feature engineering

Possible time-series features include:

```text
Rolling mean
Rolling standard deviation
Sensor slope
Sensor trend
Rate of degradation
Moving averages
Window-based statistics
```

---

## Explainable AI

Potential additions:

```text
SHAP
Permutation importance
Per-engine explanations
Local feature attribution
```

---

## Uncertainty estimation

Instead of producing only:

```text
Predicted RUL = 43.7
```

future versions could estimate:

```text
Predicted RUL = 43.7
Confidence / prediction interval
```

---

## Time-series models

More advanced approaches could use:

```text
LSTM
GRU
Temporal CNN
Transformers
Other sequence models
```

because engine telemetry is inherently sequential.

---

## Engineering improvements

Possible software-engineering additions include:

* CI/CD;
* automated testing;
* experiment tracking;
* model versioning;
* Docker;
* deployment;
* monitoring;
* automated model comparison.

---

# 34. References

## NASA

**NASA Open Data — C-MAPSS Jet Engine Simulated Data**

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

---

## Original C-MAPSS / PHM Work

Saxena, A., Goebel, K., Simon, D., & Eklund, N.

**Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation**

PHM08.

---

## Scikit-learn

**RandomForestRegressor**

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

---

## Streamlit

**Streamlit Documentation**

https://docs.streamlit.io/

---

# 35. Project Summary

## AeroSense in one diagram

```text
                    ✈️ AEROSENSE
                         │
                         ▼
              NASA C-MAPSS FD001
                         │
                         ▼
                 Engine Telemetry
                         │
                         ▼
                RUL Target Creation
                         │
                         ▼
             Engine-wise Data Split
                         │
                         ▼
              Feature Preparation
                         │
                         ▼
              Random Forest Regressor
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Validation Metrics      Test Metrics
              │                     │
              └──────────┬──────────┘
                         ▼
                Model + Metadata
                         │
                         ▼
              Feature Importance
                         │
                         ▼
                Streamlit Dashboard
                         │
                         ▼
           Interactive RUL Prediction
```

---

# 🚀 Final Takeaway

AeroSense demonstrates a complete machine-learning pipeline for predictive maintenance:

```text
Raw telemetry
     ↓
Data processing
     ↓
RUL target engineering
     ↓
Engine-wise validation
     ↓
Random Forest regression
     ↓
Model evaluation
     ↓
Feature importance
     ↓
Saved model artifacts
     ↓
Interactive Streamlit application
```

The current FD001 final-cycle test evaluation is:

```text
RMSE       = 17.660 cycles
MAE        = 13.161 cycles
R²         = 0.819
NASA Score = 498.498
```

The project is designed so that a fresh user can:

```text
Download the GitHub repository
        ↓
Install requirements
        ↓
Run model training
        ↓
Dataset downloads automatically
        ↓
Model trains
        ↓
Launch Streamlit
        ↓
Explore RUL predictions
```

No pre-existing local dataset is required for the normal first-run workflow.

---

## 🧠 AeroSense in one sentence

> **AeroSense uses NASA C-MAPSS FD001 sensor telemetry and a Random Forest regression model to estimate Remaining Useful Life, evaluate the prediction with multiple metrics, and expose the result through an interactive Streamlit dashboard.**
