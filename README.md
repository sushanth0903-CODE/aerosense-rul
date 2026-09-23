# ✈️ AeroSense

## NASA C-MAPSS Turbofan Remaining Useful Life Prediction Using Random Forest

> **AeroSense uses simulated turbofan-engine sensor telemetry and a Random Forest regression model to estimate how many operating cycles remain before an engine reaches its simulated failure point.**

AeroSense is an end-to-end **predictive-maintenance machine-learning project** built using the **NASA C-MAPSS FD001 turbofan engine dataset**.

The project demonstrates the complete journey from raw engine telemetry to an interactive Remaining Useful Life (RUL) prediction:

```text
NASA C-MAPSS FD001
        ↓
Raw Engine Telemetry
        ↓
Data Preparation
        ↓
RUL Target Construction
        ↓
Engine-wise Validation Split
        ↓
Feature Preparation
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Model + Metadata + Reports
        ↓
Interactive Streamlit Dashboard
```

The objective is not simply to train a model, but to demonstrate **how sensor data can be transformed into a meaningful machine-learning prediction and then exposed through an interactive application**.

---

# 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem](#2-the-problem)
3. [What is Remaining Useful Life?](#3-what-is-remaining-useful-life)
4. [Why RUL Prediction is Regression](#4-why-rul-prediction-is-regression)
5. [Dataset](#5-dataset)
6. [Why FD001?](#6-why-fd001)
7. [Understanding the Dataset](#7-understanding-the-dataset)
8. [Creating the RUL Target](#8-creating-the-rul-target)
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
19. [Interactive Streamlit Dashboard](#19-interactive-streamlit-dashboard)
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
31. [Project Screenshots](#31-project-screenshots)
32. [Limitations](#32-limitations)
33. [Future Improvements](#33-future-improvements)
34. [References](#34-references)
35. [Project Summary](#35-project-summary)

---

# 1. Project Overview

## 🎯 What does AeroSense do?

AeroSense estimates the **Remaining Useful Life (RUL)** of a simulated turbofan engine.

Given the engine's current telemetry:

```text
Sensor measurements
+
Operating conditions
+
Operating cycle
```

the model predicts:

```text
Estimated Remaining Useful Life
```

For example:

```text
Current cycle = 140
Predicted RUL  = 60 cycles
```

The prediction means the model estimates that approximately 60 operating cycles remain before the simulated failure point.

---

## Why is this useful?

A machine does not necessarily fail without warning.

Its sensor measurements may change as degradation progresses.

Predictive-maintenance systems attempt to use these changes to estimate remaining life **before failure occurs**.

The general idea is:

```text
Monitor
   ↓
Detect patterns
   ↓
Estimate degradation
   ↓
Predict remaining life
   ↓
Support maintenance planning
```

AeroSense demonstrates this workflow using simulated turbofan-engine data.

---

# 2. The Problem

Imagine an engine operating over hundreds of cycles.

During each cycle, multiple sensors record measurements such as temperatures, pressures, speeds, and other operating parameters.

An engine's trajectory may look conceptually like:

```text
Healthy
  │
  ▼
Normal operation
  │
  ▼
Gradual degradation
  │
  ▼
Increasing degradation
  │
  ▼
Failure
```

The machine-learning problem is:

> **Can the relationship between engine telemetry and degradation be learned well enough to estimate how many cycles remain before failure?**

AeroSense formulates this as a **supervised regression problem**.

---

# 3. What is Remaining Useful Life?

**Remaining Useful Life**, or **RUL**, is the amount of operating life remaining before an asset reaches its defined failure point.

For the C-MAPSS dataset, engine operation is measured in **cycles**.

Suppose:

```text
Failure cycle = 200
Current cycle = 140
```

Then:

```text
RUL = Failure cycle - Current cycle

RUL = 200 - 140

RUL = 60 cycles
```

So at cycle 140, the engine has:

```text
60 cycles
```

of simulated life remaining.

At the failure cycle:

```text
Current cycle = 200
Failure cycle = 200

RUL = 0
```

---

# 4. Why RUL Prediction is Regression

Machine-learning prediction can broadly be divided into tasks such as classification and regression.

### Classification

Predicts a category:

```text
Healthy
Warning
Critical
```

### Regression

Predicts a numerical value:

```text
43.7 cycles
```

AeroSense predicts:

```text
Remaining Useful Life
```

which is numerical.

Therefore:

> **AeroSense is a regression problem.**

The model used is:

```text
RandomForestRegressor
```

from scikit-learn.

---

# 5. Dataset

AeroSense uses the:

## NASA C-MAPSS Jet Engine Simulated Data

C-MAPSS stands for:

**Commercial Modular Aero-Propulsion System Simulation**

Official NASA dataset:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

The project currently uses:

```text
FD001
```

The commonly used C-MAPSS subsets are:

```text
FD001
FD002
FD003
FD004
```

AeroSense currently focuses on FD001.

---

# 6. Why FD001?

FD001 provides a relatively controlled environment for demonstrating the complete RUL-prediction pipeline.

It contains:

* multiple simulated engine trajectories;
* complete run-to-failure training trajectories;
* one operating condition;
* one degradation mode;
* multiple sensor measurements over time.

This makes FD001 a useful starting point for understanding:

```text
Telemetry
   ↓
RUL construction
   ↓
Feature preparation
   ↓
Regression
   ↓
Evaluation
```

The same overall architecture can later be extended to:

```text
FD002
FD003
FD004
```

which introduce additional operating-condition and degradation complexity.

---

# 7. Understanding the Dataset

FD001 contains:

| Property             | Value |
| -------------------- | ----: |
| Training engines     |   100 |
| Test engines         |   100 |
| Operating conditions |     1 |
| Degradation modes    |     1 |
| Operational settings |     3 |
| Sensors              |    21 |

Each raw observation contains:

```text
1 engine identifier
1 operating cycle
3 operational settings
21 sensor measurements
```

Therefore:

```text
1 + 1 + 3 + 21 = 26 raw columns
```

Conceptually, the raw data looks like:

```text
Engine | Cycle | Setting 1 | Setting 2 | Setting 3 | Sensor 1 | ... | Sensor 21
--------------------------------------------------------------------------------
1      | 1     | ...       | ...       | ...       | ...      | ... | ...
1      | 2     | ...       | ...       | ...       | ...      | ... | ...
1      | 3     | ...       | ...       | ...       | ...      | ... | ...
...
```

Each engine therefore produces a **time-ordered trajectory** of sensor observations.

---

## Dataset Files

The FD001 dataset uses three files:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

### `train_FD001.txt`

Contains complete run-to-failure trajectories.

For each training engine, observations continue until its simulated failure point.

### `test_FD001.txt`

Contains partial trajectories.

The test engines are observed only up to a final observed cycle.

### `RUL_FD001.txt`

Contains the remaining useful life of each test engine **after its final observed cycle**.

This information allows the actual final test RUL to be reconstructed.

---

# 8. Creating the RUL Target

A supervised machine-learning model needs a target value to learn.

For AeroSense, that target is:

```text
RUL
```

## Training RUL

For every training engine:

```text
RUL = Maximum cycle of that engine - Current cycle
```

Suppose an engine fails at cycle 200:

| Current Cycle | RUL |
| ------------: | --: |
|            20 | 180 |
|            50 | 150 |
|           100 | 100 |
|           150 |  50 |
|           190 |  10 |
|           200 |   0 |

Therefore:

```text
At cycle 20:

RUL = 200 - 20
    = 180
```

and:

```text
At cycle 150:

RUL = 200 - 150
    = 50
```

---

## RUL Capping

AeroSense caps the training RUL at:

```text
MAX_RUL = 125
```

The training target is therefore:

```text
RUL = min(original_RUL, 125)
```

Example:

```text
Original RUL     Training RUL
-----------      ------------
180              125
150              125
130              125
100              100
60                60
20                20
0                  0
```

This prevents very large early-life RUL values from dominating the regression problem.

The cap is applied to the **training target construction**.

---

## Test RUL Reconstruction

The test engines do not have complete trajectories.

Instead, NASA provides the remaining life after each test engine's final observed cycle.

Suppose:

```text
Final observed cycle = 150
NASA final RUL       = 30
```

Then the implied failure cycle is:

```text
150 + 30 = 180
```

For an earlier observation at cycle 100:

```text
RUL = 180 - 100
    = 80
```

AeroSense reconstructs test RUL using:

```text
RUL =
Final observed cycle
- Current cycle
+ NASA final RUL
```

For the final benchmark-style evaluation, AeroSense uses:

> **The latest observed row of each test engine.**

This produces one final prediction per test engine.

---

# 9. Complete Machine-Learning Pipeline

The complete AeroSense workflow is:

```text
NASA C-MAPSS FD001
        │
        ▼
Automatic Dataset Retrieval
        │
        ▼
Read Raw Telemetry
        │
        ▼
Create Training RUL
        │
        ▼
Reconstruct Test RUL
        │
        ▼
Engine-wise Train/Validation Split
        │
        ▼
Feature Preparation
        │
        ▼
Remove Near-Zero Variance Features
        │
        ▼
Train Random Forest
        │
        ├──────────────► Feature Importance
        │
        ▼
Validation Predictions
        │
        ▼
Validation Metrics
        │
        ▼
Final-cycle Test Predictions
        │
        ▼
Test Metrics
        │
        ▼
Save Model + Metadata + Reports
        │
        ▼
Streamlit Dashboard
```

In simple terms:

> **Data → Target → Features → Model → Prediction → Evaluation → Dashboard**

---

# 10. Feature Preparation

The raw dataset contains:

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

as a predictive feature.

Why?

Because it is an identifier:

```text
Engine 1
Engine 2
Engine 3
...
```

It does not represent the physical state of the engine.

The target:

```text
RUL
```

is also excluded from the model inputs because it is the value being predicted.

Therefore:

```text
Excluded:
    unit_number
    RUL

Used:
    cycle
    operational settings
    informative sensors
```

---

## Near-Zero Variance Features

A feature that barely changes provides little information.

For example:

```text
10
10
10
10
10
```

has essentially no variation.

AeroSense calculates feature variance from the training data and removes features whose variance is effectively zero:

```text
variance > 1e-12
```

This prevents constant-like features from being unnecessarily passed to the model.

---

# 11. Preventing Data Leakage

Data leakage is one of the most important concerns in this type of dataset.

Each engine produces many observations:

```text
Engine 1 → many cycles
Engine 2 → many cycles
Engine 3 → many cycles
...
```

A naive row-level random split could place observations from the **same engine** in both training and validation.

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

The model would already have seen part of the same engine's trajectory.

That can make validation performance look artificially optimistic.

---

## Engine-wise Split

AeroSense uses:

```text
GroupShuffleSplit
```

with the engine identifier as the grouping variable.

The split is performed at the **engine level**, not the row level.

Conceptually:

```text
Engine 1 → Training
Engine 2 → Training
Engine 3 → Validation
Engine 4 → Training
Engine 5 → Validation
...
```

A complete engine belongs to only one split.

The current configuration uses:

```text
Validation fraction = 20%
random_state = 42
```

This creates a more meaningful evaluation of performance on previously unseen engine trajectories.

---

# 12. Why Random Forest?

AeroSense uses:

```text
Random Forest Regressor
```

because it is a practical baseline for numerical/tabular telemetry data.

Important characteristics include:

### Non-linear relationships

Engine degradation does not necessarily follow a simple straight-line relationship.

Random Forest can learn nonlinear relationships.

### Feature interactions

Multiple sensors can interact in ways that are difficult to represent using a simple linear equation.

Tree-based models can learn these interactions.

### No feature scaling requirement

Random Forest does not require the standardization or normalization commonly needed by some other machine-learning algorithms.

### Suitable for tabular data

The C-MAPSS dataset is structured numerical/tabular data, which is a natural setting for tree-based models.

### Feature importance

Random Forest provides feature-importance information that can be visualized.

### Interpretable model structure

Individual decision trees can be explained using threshold-based rules, making the model easier to inspect than many more complex approaches.

---

# 13. How Random Forest Works

A **decision tree** predicts by repeatedly splitting data using learned conditions.

Conceptually:

```text
Is sensor_7 < 42.5?
          │
      ┌───┴───┐
      │       │
     Yes      No
      │       │
      ▼       ▼
   Branch A  Branch B
```

A Random Forest combines many decision trees.

For example:

```text
Tree 1   → 42 cycles
Tree 2   → 48 cycles
Tree 3   → 45 cycles
Tree 4   → 41 cycles
...
Tree 100 → 46 cycles
```

The forest combines these predictions to produce the final regression output.

Conceptually:

```text
                 ┌── Tree 1 ──┐
                 ├── Tree 2 ──┤
Input telemetry ├── Tree 3 ──┼──► Combined RUL
                 ├──   ...  ──┤
                 └── Tree 100 ┘
```

This is called **ensemble learning**.

Instead of relying on one tree, the model combines many trees.

---

# 14. Random Forest Configuration

The current model configuration is:

```text
n_estimators      = 100
max_depth         = 15
min_samples_split = 5
random_state      = 42
n_jobs            = -1
oob_score         = True
```

### `n_estimators = 100`

The forest contains:

```text
100 decision trees
```

### `max_depth = 15`

Each tree can grow to a maximum depth of 15.

This limits individual tree complexity.

### `min_samples_split = 5`

A node must contain at least five samples before it can be split.

### `random_state = 42`

Fixes the random seed for reproducibility.

### `n_jobs = -1`

Allows scikit-learn to use all available CPU cores.

### `oob_score = True`

Enables out-of-bag evaluation within the Random Forest.

---

# 15. Why Feature Scaling is Not Used

Some machine-learning algorithms are sensitive to feature scale.

For example:

```text
Sensor A: 0.1 → 1.2
Sensor B: 1000 → 5000
```

Distance-based algorithms can be affected by these different ranges.

Random Forest works differently.

Decision trees primarily make threshold-based decisions such as:

```text
Is sensor_7 < 42.5?
```

The model does not calculate distances between samples in the way algorithms such as k-nearest neighbors do.

Therefore, explicit normalization or standardization is **not required** for this Random Forest pipeline.

The original numerical meaning of the sensor measurements can therefore be preserved.

---

# 16. Model Evaluation

AeroSense uses four evaluation metrics:

```text
RMSE
MAE
R²
NASA Asymmetric Score
```

Each describes a different aspect of model performance.

---

## 16.1 RMSE

**Root Mean Squared Error**

RMSE gives greater influence to larger errors.

Conceptually:

```text
Prediction errors
       ↓
Square each error
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

**Mean Absolute Error**

MAE measures the average absolute difference between predicted and actual RUL.

Example:

```text
Actual RUL    = 50
Predicted RUL = 43

Absolute error = 7 cycles
```

Therefore:

> **MAE answers approximately how many cycles away the predictions are from the actual values on average.**

Lower MAE indicates smaller average error.

---

## 16.3 R²

**Coefficient of Determination**

R² measures how well the model explains variation in the target compared with a mean-based baseline.

Generally:

```text
R² closer to 1
       ↓
Stronger explanatory performance
```

R² is **not an accuracy percentage**.

For example:

```text
R² = 0.819
```

does not mean:

```text
81.9% prediction accuracy
```

---

## 16.4 NASA Asymmetric Score

RUL prediction errors can have different penalties depending on their direction.

AeroSense therefore calculates the asymmetric scoring function commonly associated with C-MAPSS / PHM RUL evaluation.

Let:

```text
d = predicted_RUL - actual_RUL
```

Then the score contribution is:

```text
if d < 0:

    exp(-d / 13) - 1

else:

    exp(d / 10) - 1
```

The total score is the sum of the contributions.

Important interpretation:

```text
Perfect predictions → 0
Lower score        → better
```

The two directions of error are penalized differently.

---

# 17. Current Results

The current AeroSense Random Forest model produced the following **final-cycle FD001 test results**:

| Metric         |            Result |
| -------------- | ----------------: |
| **RMSE**       | **17.660 cycles** |
| **MAE**        | **13.161 cycles** |
| **R²**         |         **0.819** |
| **NASA Score** |       **498.498** |

The current engine-wise validation results are:

| Metric                    |            Result |
| ------------------------- | ----------------: |
| **Validation RMSE**       | **15.999 cycles** |
| **Validation MAE**        | **10.840 cycles** |
| **Validation R²**         |         **0.853** |
| **Validation NASA Score** |     **22627.066** |

These values correspond to the current implementation, Random Forest configuration, feature-processing pipeline, and FD001 dataset.

They should not be interpreted as guarantees of performance on real aircraft engines.

---

# 18. Feature Importance

Random Forest provides feature-importance values based on the trained ensemble.

AeroSense saves these values to:

```text
reports/feature_importance.csv
```

The dashboard visualizes the most important features.

This answers:

> **Which telemetry features did the trained model rely on most strongly?**

The five most important sensor features are also exposed through the dashboard for interactive exploration.

---

## Important distinction

Feature importance does **not** automatically mean:

```text
"This sensor causes engine degradation."
```

Instead, it means:

```text
"The trained model relied more heavily on this feature
relative to other features during its learned tree splits."
```

Model importance, correlation, and physical causation are different concepts.

---

# 19. Interactive Streamlit Dashboard

AeroSense includes an interactive dashboard built with:

```text
Streamlit
```

The dashboard connects the trained model to a visual interface.

```text
Select Engine
      ↓
Load Latest Telemetry
      ↓
Generate Baseline Prediction
      ↓
Modify Important Sensors
      ↓
Generate Adjusted Prediction
      ↓
Compare Predictions
```

---

## Engine Selection

The dashboard allows a test engine to be selected by its engine ID.

---

## Latest Telemetry

For the selected engine, the dashboard loads its:

```text
latest observed cycle
```

This becomes the baseline input to the model.

---

## Baseline Prediction

The model predicts RUL using the engine's actual latest telemetry.

This is displayed as the:

```text
Baseline Prediction
```

---

## Interactive Sensor Controls

The five most important sensor features can be modified using sliders.

This creates an educational **what-if experiment**:

```text
Original telemetry
       ↓
Modify sensor value
       ↓
Run Random Forest again
       ↓
Observe new RUL prediction
```

The purpose is to make the model's response to its inputs visible.

The modified values are artificial exploration inputs; they do not represent a physically validated future engine state.

---

## RUL Gauge

The dashboard displays the predicted RUL using a visual gauge.

The value is expressed in:

```text
operating cycles
```

---

## Baseline vs Adjusted Prediction

The dashboard compares:

```text
Original prediction
        vs
Modified prediction
```

and displays the change:

```text
Adjusted prediction - Baseline prediction
```

This makes the effect of the interactive sensor modifications easier to observe.

---

## Feature Importance

The dashboard displays the model's most important features.

---

## Telemetry Visualization

The dashboard visualizes the adjustable sensor values and their current states.

---

## Sensor Information

The dashboard provides information such as:

* sensor identifier;
* engineering symbol;
* current value;
* unit;
* physical meaning.

---

## Raw Telemetry

An expandable section provides access to the underlying telemetry for the selected engine.

---

## Built-in Explanation

The dashboard also explains the main project concepts, including:

* RUL;
* Random Forest;
* model evaluation;
* sensor interaction;
* dashboard interpretation.

---

# 20. Dashboard Status Categories

For visualization purposes, AeroSense groups predicted RUL into three categories:

|    Predicted RUL | Dashboard Status       |
| ---------------: | ---------------------- |
|  **> 60 cycles** | 🟢 SAFE                |
| **21–60 cycles** | 🟡 MAINTENANCE WARNING |
|  **≤ 20 cycles** | 🔴 CRITICAL            |

### ⚠️ Important

These are **project-defined visualization categories**.

They are:

* not official aviation thresholds;
* not certified maintenance limits;
* not safety classifications;
* not real aircraft maintenance recommendations.

---

# 21. Project Architecture

AeroSense separates the major responsibilities of the system:

```text
                    ┌──────────────────────────┐
                    │ NASA C-MAPSS FD001 Data │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     data_pipeline.py     │
                    │                          │
                    │ • Dataset acquisition    │
                    │ • Data loading            │
                    │ • RUL construction        │
                    │ • Test RUL reconstruction│
                    │ • Engine-wise splitting   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      model_train.py      │
                    │                          │
                    │ • Feature preparation    │
                    │ • Random Forest training │
                    │ • Prediction              │
                    │ • Artifact generation    │
                    └───────┬──────────┬───────┘
                            │          │
              ┌─────────────┘          └─────────────┐
              ▼                                      ▼
┌──────────────────────────┐            ┌──────────────────────────┐
│       evaluate.py        │            │         reports          │
│                          │            │                          │
│ • RMSE                   │            │ • metrics.json           │
│ • MAE                    │            │ • feature_importance.csv │
│ • R²                     │            │ • test_predictions.csv   │
│ • NASA score             │            └──────────────────────────┘
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│               Streamlit Dashboard                  │
│                                                    │
│ Engine → Telemetry → Prediction → Gauge →         │
│ Feature Importance → Sensor Interaction            │
└────────────────────────────────────────────────────┘
```

---

# 22. Repository Structure

The GitHub repository contains:

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

The generated dataset, trained model, and reports are excluded from Git using `.gitignore`.

---

# 23. What Each File Does

## `src/data_pipeline.py`

The **data-processing layer**.

Responsible for:

* checking whether FD001 exists;
* downloading the dataset when necessary;
* handling the archive structure;
* extracting the required FD001 files;
* loading raw telemetry;
* assigning column names;
* calculating training RUL;
* reconstructing test RUL;
* preparing engine-wise validation data;
* identifying informative features.

---

## `src/model_train.py`

The **model-training layer**.

Responsible for:

1. loading FD001;
2. creating the engine-wise validation split;
3. selecting model features;
4. removing near-zero variance features;
5. creating the Random Forest;
6. training the model;
7. generating validation predictions;
8. evaluating final-cycle test predictions;
9. calculating feature importance;
10. saving the trained model;
11. saving metadata;
12. saving reports.

---

## `src/evaluate.py`

The **evaluation layer**.

Provides:

* RMSE;
* MAE;
* R²;
* NASA asymmetric score;
* combined metric calculations;
* metric-report generation;
* dashboard status logic.

---

## `app/streamlit_app.py`

The **application layer**.

Responsible for:

* loading the trained model;
* loading metadata;
* loading test telemetry;
* engine selection;
* baseline predictions;
* sensor controls;
* adjusted predictions;
* prediction comparison;
* RUL gauge;
* feature importance;
* telemetry visualization;
* sensor information.

---

## `tests/test_data_pipeline.py`

Contains automated tests for important data-processing behavior, including RUL-related functionality.

---

## `tests/test_evaluate.py`

Contains automated tests for the evaluation functions and scoring logic.

---

## `PROJECT_NOTES.md`

Contains technical implementation and design notes, including important decisions about:

* model selection;
* feature scaling;
* engine-wise validation;
* dataset handling;
* evaluation;
* dashboard thresholds.

---

# 24. Installation

## Requirements

AeroSense uses:

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

Exact dependency versions are pinned in:

```text
requirements.txt
```

The current tested environment uses:

```text
Python 3.14
```

---

## Clone the repository

```bash
git clone https://github.com/sushanth0903-CODE/aerosense-rul.git
cd aerosense-rul
```

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

AeroSense does **not require the NASA dataset to already exist locally**.

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

If they are missing, the pipeline automatically:

```text
1. Downloads the C-MAPSS archive
        ↓
2. Opens the downloaded archive
        ↓
3. Searches for the required FD001 files
        ↓
4. Handles nested ZIP structures
        ↓
5. Extracts the required files
        ↓
6. Continues with training
```

Therefore, a normal fresh setup only requires:

```bash
py -m src.model_train
```

on Windows.

A successful first run will display messages similar to:

```text
NASA C-MAPSS archive not found. Downloading it now...
NASA C-MAPSS FD001 dataset is ready.
```

After the first successful download, the local dataset is reused.

---

## Manual fallback

Automatic dataset retrieval requires internet access.

If the dataset host is inaccessible, the required FD001 files can be downloaded from the official NASA dataset page:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Place:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

inside:

```text
data/raw/
```

This is only a fallback; manual preparation is not required during normal operation.

---

# 26. Train the Model

From the repository root:

### Windows

```bash
py -m src.model_train
```

### macOS / Linux

```bash
python -m src.model_train
```

The command performs the complete workflow:

```text
Check dataset
      ↓
Download if necessary
      ↓
Load FD001
      ↓
Build RUL targets
      ↓
Split engines
      ↓
Prepare features
      ↓
Train Random Forest
      ↓
Evaluate validation data
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

After training has completed:

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

Open the displayed address in a browser.

---

# 28. Run the Tests

AeroSense includes automated tests for core functionality.

### Windows

```bash
py -m pytest
```

### macOS / Linux

```bash
python -m pytest
```

Current local result:

```text
3 passed
```

The tests cover important data-processing and evaluation behavior.

---

# 29. Generated Files

Running the training pipeline generates three categories of local data.

## Dataset

```text
data/raw/
```

Contains the downloaded FD001 dataset.

---

## Model Artifacts

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

Contains information required by the application, including:

* selected features;
* important sensor features;
* sensor information;
* feature ranges;
* RUL configuration;
* model parameters.

---

## Reports

```text
reports/metrics.json
reports/feature_importance.csv
reports/test_predictions.csv
```

### `metrics.json`

Stores validation and test evaluation information.

### `feature_importance.csv`

Stores the model's feature-importance values.

### `test_predictions.csv`

Contains final test predictions and related information such as:

```text
engine number
final observed cycle
actual RUL
predicted RUL
absolute error
```

---

# 30. Reproducibility

AeroSense uses explicit configuration values to make the experiment reproducible.

The Random Forest uses:

```text
random_state = 42
```

The model configuration is defined in:

```text
src/model_train.py
```

Model metadata is saved to:

```text
artifacts/metadata.joblib
```

This allows the dashboard to use the same feature definitions and model configuration as the training pipeline.

---

## Fresh reproduction

A complete fresh setup can be performed using:

```bash
git clone https://github.com/sushanth0903-CODE/aerosense-rul.git
cd aerosense-rul

py -m venv .venv
.venv\Scripts\activate

py -m pip install -r requirements.txt

py -m src.model_train

py -m streamlit run app/streamlit_app.py
```

The dataset is downloaded automatically during the first training run when it is not already present.

---

# 31. Project Screenshots

The repository can include screenshots of the main application components.

## AeroSense Dashboard

![AeroSense Dashboard](screenshots/dashboard.png)

## Interactive Sensor Controls

![Sensor Interaction](screenshots/sensor-interaction.png)

## Random Forest Feature Importance

![Feature Importance](screenshots/feature-importance.png)

## Model Evaluation

![Model Evaluation](screenshots/model-evaluation.png)

These screenshots provide a quick visual overview of the application's interface and outputs.

---

# 32. Limitations

AeroSense is a machine-learning project based on simulated engine data.

Several limitations should be considered.

## 1. Simulated dataset

C-MAPSS is a simulated turbofan-engine dataset.

Real aircraft telemetry can contain:

* measurement noise;
* missing values;
* sensor faults;
* maintenance events;
* changing operating conditions;
* environmental effects;
* operational variation.

These factors are not completely represented by the current project.

---

## 2. FD001 only

The current implementation focuses on:

```text
FD001
```

The remaining C-MAPSS subsets are not currently part of the main pipeline.

---

## 3. Single baseline model

The project currently uses:

```text
Random Forest
```

A broader experiment could compare several machine-learning approaches.

---

## 4. No uncertainty estimation

The model currently produces a point prediction such as:

```text
43.7 cycles
```

It does not provide a calibrated prediction interval.

---

## 5. Dashboard sensor modifications are hypothetical

The interactive sliders modify sensor values artificially.

They demonstrate model sensitivity but do not simulate a physically validated future engine condition.

---

## 6. Dashboard thresholds are not maintenance rules

The:

```text
SAFE
MAINTENANCE WARNING
CRITICAL
```

categories are interface labels created for demonstration.

They are not certified aviation maintenance limits.

---

## 7. Real-world deployment is not established

Performance on C-MAPSS does not demonstrate equivalent performance on real aircraft engines.

A real deployment would require extensive:

* real-world validation;
* safety analysis;
* uncertainty estimation;
* domain expertise;
* monitoring;
* testing;
* regulatory compliance.

---

# 33. Future Improvements

AeroSense can be extended in several directions.

## Additional C-MAPSS subsets

Support:

```text
FD001
FD002
FD003
FD004
```

This would introduce more varied operating conditions and degradation scenarios.

---

## Additional machine-learning models

Possible comparisons include:

```text
Random Forest
Extra Trees
Gradient Boosting
XGBoost
LightGBM
```

---

## Hyperparameter optimization

The fixed baseline configuration could be investigated using:

```text
Grid Search
Random Search
Bayesian Optimization
```

---

## Feature engineering

Potential time-series features include:

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

Possible additions include:

```text
SHAP
Permutation Importance
Local Explanations
Per-engine Feature Attribution
```

---

## Uncertainty estimation

Instead of producing only:

```text
Predicted RUL = 43.7
```

future versions could estimate:

```text
Predicted RUL
+
Prediction Interval
```

---

## Time-series models

Because engine telemetry is sequential, future versions could investigate:

```text
LSTM
GRU
Temporal CNN
Transformers
Other sequence models
```

---

## Engineering improvements

The project could also be extended with:

* continuous integration;
* broader automated testing;
* experiment tracking;
* model versioning;
* Docker;
* deployment;
* monitoring;
* automated model comparison.

---

# 34. References

## NASA C-MAPSS Dataset

**NASA Open Data — C-MAPSS Jet Engine Simulated Data**

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

---

## C-MAPSS / PHM Reference

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

AeroSense can be summarized as:

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

## Current FD001 Test Result

```text
RMSE       = 17.660 cycles
MAE        = 13.161 cycles
R²         = 0.819
NASA Score = 498.498
```

The complete workflow is:

```text
Raw telemetry
      ↓
RUL target engineering
      ↓
Engine-wise validation
      ↓
Feature preparation
      ↓
Random Forest regression
      ↓
Prediction
      ↓
Evaluation
      ↓
Feature importance
      ↓
Saved artifacts
      ↓
Interactive dashboard
```

A fresh user can reproduce the project by:

```text
Clone repository
      ↓
Install dependencies
      ↓
Run training
      ↓
Dataset downloads automatically
      ↓
Model trains
      ↓
Reports are generated
      ↓
Launch Streamlit
      ↓
Explore RUL predictions
```

No pre-existing local NASA dataset is required for the normal first-run workflow.

---

## 🧠 AeroSense in One Sentence

> **AeroSense uses NASA C-MAPSS FD001 sensor telemetry and a Random Forest regression model to estimate Remaining Useful Life, evaluate the predictions using multiple metrics, and expose the trained system through an interactive Streamlit dashboard.**
