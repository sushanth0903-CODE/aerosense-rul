# AeroSense ✈️

### NASA C-MAPSS Turbofan Degradation & Remaining Useful Life Predictor using Random Forest

AeroSense is a first-year-friendly but technically serious **predictive maintenance** project. It uses a **Random Forest Regressor** to estimate the **Remaining Useful Life (RUL)** of simulated turbofan engines from sensor telemetry.

The project is built around NASA's C-MAPSS FD001 dataset. NASA describes FD001 as having 100 training trajectories, 100 test trajectories, one sea-level operating condition, and one HPC degradation fault mode. The raw files contain 26 space-separated columns: engine ID, cycle, 3 operating settings, and 21 sensor measurements. The test set is accompanied by one true final RUL value per test engine. 

## 1. What problem are we solving?

Imagine an aircraft engine is running and its sensors are continuously producing measurements. Instead of waiting for a failure, we want to answer:

> **"Approximately how many operating cycles does this engine have left?"**

That number is the **Remaining Useful Life**.

This is a **regression** problem because the target is a continuous number of cycles, not a class such as healthy/faulty.

## 2. Why Random Forest?

Random Forest is an ensemble of decision trees. Each tree learns a set of if/then splits, and the forest averages the predictions from many trees.

It is a strong fit for this project because:

- it captures non-linear relationships between sensors and RUL;
- it needs little preprocessing for numerical tabular data;
- it does **not require feature normalization/scaling**;
- it provides feature-importance values that make the project easier to interpret.

## 3. Dataset

NASA C-MAPSS FD001 is a simulated turbofan run-to-failure dataset. Training engines continue to failure, while test engines stop before failure and receive a separate final-RUL truth file.

Official NASA dataset page:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

NASA's current catalog lists the resource as public; its landing page notes that the dataset license is not specified. For that reason, this repository **does not redistribute the raw dataset**. The code downloads it at runtime instead.

## 4. RUL target creation

For a training engine:

```text
RUL = maximum cycle of that engine - current cycle
```

Example:

```text
engine reaches cycle 200
current cycle = 140
RUL = 200 - 140 = 60 cycles
```

The training target is capped at **125 cycles**. This prevents very early, relatively uninformative rows from dominating the regression target.

For test data, NASA supplies one final RUL value per engine. AeroSense reconstructs the row-level test RUL as:

```text
RUL = final observed test cycle - current cycle + final_RUL
```

For benchmark-style test evaluation, we use **the final observed row of each test engine**, because that is where the supplied test RUL corresponds to a prediction point.

## 5. Project architecture

```text
aerosense-rul/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   └── .gitkeep
├── data/
│   ├── README.md
│   └── raw/
│       └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── evaluate.py
│   └── model_train.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## 6. Pipeline

```text
NASA C-MAPSS FD001
        │
        ▼
Download + parse
        │
        ▼
Create RUL target
        │
        ▼
Engine-wise train/validation split
        │
        ▼
Remove constant / near-constant features
        │
        ▼
Random Forest Regressor
        │
        ├──────────────► Feature importance
        │
        ▼
Validation + final test evaluation
        │
        ▼
Saved model artifacts
        │
        ▼
Streamlit dashboard
```

## 7. Leakage control

A random row split would allow rows from the same engine to appear in both train and validation. Since an engine produces a time series, that can make validation look artificially easy.

AeroSense therefore splits by **whole engine ID** using `GroupShuffleSplit`. Entire engines are kept out of the validation set.

The engine ID itself is also excluded from model features because it is an identifier, not a physical measurement.

## 8. Model configuration

The baseline Random Forest uses:

```text
n_estimators      = 100
max_depth         = 15
min_samples_split = 5
random_state      = 42
n_jobs             = -1
oob_score          = True
```

These values are intentionally fixed so that the core algorithm remains easy to explain in a technical interview.

## 9. Evaluation metrics

### RMSE

Penalizes large errors more strongly than MAE.

### MAE

Average absolute prediction error in cycles.

### R²

Measures how much of the variance in the target is explained by the model.

### NASA / PHM asymmetric score

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

The benchmark intentionally treats early and late RUL errors differently. A lower total score is better, with a perfect predictor scoring 0.

## 10. Streamlit dashboard

The dashboard lets you:

1. select a test engine;
2. start from its latest available telemetry;
3. modify the five most important sensor measurements;
4. see the predicted RUL update immediately;
5. inspect the model's top feature importances.

The UI uses three **demo** states:

| Predicted RUL | Demo status |
|---:|---|
| > 60 cycles | Safe |
| 21–60 cycles | Maintenance Warning |
| ≤ 20 cycles | Critical |

These thresholds are **not aviation-certified maintenance limits**. They are presentation categories for an educational project.

## 11. Installation

Use Python 3.10+.

```bash
git clone <YOUR-REPOSITORY-URL>
cd aerosense-rul
python -m venv .venv
```

Activate the environment.

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## 12. Train the model

From the project root:

```bash
python -m src.model_train
```

The first run downloads the NASA dataset into `data/raw/`, trains the model, and creates:

```text
artifacts/model.joblib
artifacts/metadata.joblib
reports/metrics.json
reports/feature_importance.csv
reports/test_predictions.csv
```

The generated model/report files are ignored by Git because they can be reproduced from the source code and dataset.

## 13. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

## 14. What each file does

### `src/data_pipeline.py`

Downloads the dataset, parses the raw text files, computes RUL, and prepares FD001.

### `src/evaluate.py`

Contains RMSE, MAE, R², the asymmetric NASA score, and dashboard status logic.

### `src/model_train.py`

Creates the engine-wise split, trains Random Forest, evaluates it, saves the model, and exports feature importance.

### `app/streamlit_app.py`

Loads the saved model and provides an interactive telemetry/RUL dashboard.

## References

- NASA Open Data, **CMAPSS Jet Engine Simulated Data**: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Saxena, A., Goebel, K., Simon, D., & N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08.
