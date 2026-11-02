# Customer Churn Prediction & Monitoring

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)

A Python customer churn project covering **model training, batch prediction, a FastAPI prediction service, data drift detection, and performance evaluation**.

The main project turns churn probabilities into ranked customer contact lists, compares contact strategies, and explores how simulated feature drift affects model performance. Supporting assignments document earlier supervised learning exercises.

[Quick Start](#quick-start) · [Prediction API](#prediction-api) · [Sample Results](#sample-results) · [Tests](#tests) · [Repository Guide](#repository-guide)

## Features

- **Training pipeline:** logistic regression with numeric and categorical preprocessing, saved as a complete joblib pipeline.
- **Batch scoring:** customer-level churn probabilities, descending risk ranks, and a configurable contact capacity.
- **Prediction API:** single and batch predictions with Pydantic request validation.
- **Input checks:** required columns, unique customer IDs, numeric constraints, and missing-value handling.
- **Drift detection:** Kolmogorov–Smirnov tests for continuous features and chi-square tests for categorical features, with Benjamini–Hochberg correction.
- **Performance monitoring:** ROC-AUC, log loss, Brier score, precision, recall, F1, and a simplified campaign value calculation.
- **API tests:** mocked inference to verify prediction responses and validation behavior without loading the saved model.

## Model & Data

The main workflow uses the bundled [`Telco-Customer-Churn.csv`](supervised_learning_project/data/Telco-Customer-Churn.csv) dataset. The target is encoded as `Churn = 1` for churn and `0` otherwise; `customerID` is retained for matching outputs and excluded from training features.

| Component | Implementation |
| --- | --- |
| Numeric inputs | Tenure, monthly charges, total charges, senior citizen flag |
| Categorical inputs | Contract, payment method, demographics, and subscribed services |
| Numeric preprocessing | Median imputation and standard scaling |
| Categorical preprocessing | Most-frequent imputation and one-hot encoding |
| Estimator | `LogisticRegression(max_iter=5000)` |
| Split | Stratified 80/20 train/test split, `random_state=42` |

Blank total charges are set to zero for new customers with zero tenure. Other missing values are handled by the preprocessing pipeline.

## Quick Start

Use **Python 3.10 or newer**.

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/noname1134/ml-supervised-learning.git
cd ml-supervised-learning
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use `py -m venv .venv` and activate with `.venv\Scripts\Activate.ps1`.

### 2. Install dependencies

```bash
python -m pip install numpy pandas scikit-learn scipy statsmodels joblib fastapi "pydantic>=2" uvicorn pytest httpx
```

### 3. Enter the main project directory

```bash
cd supervised_learning_project
```

**Run all remaining commands from this directory.** The scripts resolve data, model, and report paths relative to the working directory.

### 4. Run the complete workflow

Run these commands in order:

```bash
python assignment_49_train.py
python assignment_49_predict.py
python assignment_50_monitor.py
python assignment_49_predict.py --input test/drifted_test_customers.csv --output result/drifted_churn_model_predictions.csv
python assignment_51_performance_monitor.py
```

| Step | Output |
| --- | --- |
| Train | Fitted model, feature names, reference training data, test customers, and held-out labels |
| Predict | Ranked customer campaign CSV, selecting up to 125 customers by default |
| Detect drift | Original and simulated drift reports, plus a simulated customer batch |
| Score simulated batch | Churn predictions for the shifted feature data |
| Evaluate | Performance comparison for original and simulated data, using threshold and top-N strategies |

Running the workflow overwrites the corresponding generated artifacts under `models/`, `data/`, `test/`, and `result/`.

To change the contact capacity:

```bash
python assignment_49_predict.py --capacity 100 --output result/campaign_100.csv
```

The campaign output contains `customerID`, `churn_probability`, `customer_rank`, and `selected_for_contact`. Additional arguments are documented by each script's `--help` option, except the training script, which uses configured paths.

## Prediction API

Start the API from `supervised_learning_project/`:

```bash
python -m uvicorn api.main:app --reload
```

Open [Swagger UI](http://127.0.0.1:8000/docs) to inspect the schemas and submit requests. The saved model is loaded during application startup; retrain it locally if the committed artifact is incompatible with your installed scikit-learn version.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Return service health status |
| `POST` | `/predict` | Return one customer's churn probability |
| `POST` | `/predict/batch` | Predict for 1–1,000 customers with unique IDs |
| `POST` | `/customers/validate` | Validate a minimal customer payload |
| `GET` | `/customers/{customer_id}` | Demonstration endpoint returning the requested ID; no database lookup |

### Single prediction request

Send this JSON to `POST /predict`:

```json
{
  "customerID": "DEMO-001",
  "tenure": 1,
  "MonthlyCharges": 29.85,
  "Contract": "Month-to-month",
  "TotalCharges": 29.85,
  "SeniorCitizen": 0,
  "gender": "Female",
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "No",
  "PaperlessBilling": "Yes",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "PaymentMethod": "Electronic check",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No"
}
```

The response contains `customer_id` and `churn_probability`, a number between 0 and 1. For batch prediction, wrap full customer objects in a `customers` array. Field names are case-sensitive; the minimal validation endpoint uses a different schema shown in Swagger UI.

## Drift Monitoring

The monitoring script compares reference training features with the held-out customer batch and a simulated variant. The simulation increases monthly charges by 30, halves tenure, sets contracts to month-to-month, and introduces missing payment methods for approximately 30% of customers.

Reports include test statistics, raw and adjusted p-values, effect sizes, missing-value rates, and drift flags at a default significance level of 0.05. Numeric effect sizes use the KS statistic; categorical effect sizes use Cramér's V.

## Sample Results

The committed [performance report](supervised_learning_project/result/performance_evaluation.csv) contains these results. They are existing sample outputs, not a fresh benchmark run.

| Data | ROC-AUC | Log loss | Brier score | Precision, top 125 | Churners in top 125 |
| --- | --- | --- | --- | --- | --- |
| Original | 0.8420 | 0.4204 | 0.1381 | 0.736 | 92 |
| Simulated drift | 0.7739 | 0.4955 | 0.1658 | 0.608 | 76 |

The report also compares a **0.5 probability threshold** with a **top-125 contact strategy**. Its `net_profit` field uses an illustrative assumption of 500 value units per churner identified and 50 cost units per contact. This is a simplified scoring formula, not measured retention revenue.

Simulated feature changes retain the original held-out labels. The comparison illustrates sensitivity to an artificial distribution shift rather than performance on a newly observed customer population.

## Tests

```bash
python -m pytest api_tests/ -v
```

The suite covers health responses, customer validation, single and batch prediction, duplicate IDs, empty batches, invalid customers, and the batch size limit. A deterministic mock model replaces model loading, so these tests check API behavior rather than predictive quality.

## Repository Guide

| Path | Contents |
| --- | --- |
| [`supervised_learning_project/`](supervised_learning_project/) | Main churn prediction and monitoring workflow |
| [`assignments/`](assignments/) | Earlier exercises on classification, cross-validation, model selection, calibration, and decision thresholds |
| `supervised_learning_project/api/` | FastAPI routes and inference service |
| `supervised_learning_project/api_tests/` | API tests and mock-model fixtures |
| `supervised_learning_project/utils.py` | Shared paths, data validation, CSV handling, and cleaning |
| `supervised_learning_project/models/` | Saved pipeline and expected feature names |
| `supervised_learning_project/result/` | Prediction, drift, and performance CSV reports |

## Scope

This is a local educational project with batch monitoring scripts and a development API. Monitoring is run manually; automated retraining and production deployment are outside the current implementation.

## Author

[Liad Illouz](https://github.com/noname1134)
