# intelligent-umrah-crowd-forecasting-system

A Data Science graduation project that predicts Umrah visitor crowd levels and recommends suitable visiting days at Al-Masjid Al-Haram. The system integrates a Growth-Rate XGBoost forecasting model with an interactive Arabic-language Streamlit dashboard deployed for public access.

---

## Live Dashboard

https://intelligent-umrah-crowd-forecasting-system-xdui7hdtsbfwcm6tjhw.streamlit.app/

---

## Project Overview

This system addresses the challenge of crowd management at one of the world's most visited religious sites. By analyzing historical Umrah visitor data aligned with the Hijri calendar, the system produces day-level crowd forecasts and translates them into actionable visiting recommendations.

Users interact with the dashboard by selecting a Hijri month and day. The system returns the predicted visitor count, a crowd level classification, a visiting recommendation, and an alternative lower-crowd day within a 7-day window — supported by an interactive forecast visualization and a detailed results table.

---

## Project Layers

### 1. Modeling Layer

The modeling pipeline was developed in `notebooks/final_model_notebook.ipynb` and productionized in `train_model.py`. It covers the following stages:

- Data collection, preprocessing, and cleaning
- Hijri calendar alignment
- Weather feature integration
- Feature engineering, including lag-based and rolling-window features
- Growth-rate target construction (`Predicted_Growth_Rate_7`)
- XGBoost model training on Hijri year 1445H
- Model evaluation and testing on Hijri year 1446H
- Forecast result generation saved to `growth_rate_xgboost_results_1446.xlsx`

### 2. Dashboard and Deployment Layer

The Streamlit dashboard (`app.py`) serves as the user-facing interface. It loads the trained model artifact (`growth_rate_xgboost_model.pkl`) and the precomputed forecast results to display interactive forecasts and recommendations based on the selected Hijri month and day.

The dashboard is deployed through Streamlit Community Cloud and is publicly accessible via the link above.

---

## Repository Structure

```
intelligent-umrah-crowd-forecasting-system/
├── app.py
├── train_model.py
├── README.md
├── requirements.txt
├── runtime.txt
├── growth_rate_xgboost_model.pkl
├── growth_rate_xgboost_results_1446.xlsx
├── Intelligent System for Predicting Visitor Crowd Levels and Optimal Visiting Times at Al-Masjid Al-Haram.xlsx
│
├── notebooks/
│   └── final_model_notebook.ipynb
│
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
│
├── reports/
├── poster/
└── presentation/
```

---

## Data Organization

| Folder | Description |
|---|---|
| `data/raw/` | Original source datasets used during preprocessing and feature engineering |
| `data/external/` | Supporting external files, including Hijri calendar data for 1445H and 1446H |
| `data/processed/` | Final processed dataset used for model training and dashboard deployment |

The processed dataset is additionally stored in the repository root, as the Streamlit application and training script resolve the file path from that location.

The `reports/`, `poster/`, and `presentation/` folders are designated for official GP2 supplementary deliverables.

---

## Model

**Architecture:** Growth-Rate XGBoost

The model predicts a 7-day forward growth rate:

```
Predicted_Growth_Rate_7
```

The final daily visitor count is computed from the 7-day lag value as follows:

```
Prediction = lag_7 * (1 + Predicted_Growth_Rate_7)
```

| Artifact | Description |
|---|---|
| `growth_rate_xgboost_model.pkl` | Serialized trained model loaded at runtime by the dashboard |
| `growth_rate_xgboost_results_1446.xlsx` | Precomputed forecast results for Hijri year 1446H |

---

## Dashboard Features

The Streamlit dashboard provides the following functionality:

- Arabic-language user interface
- Hijri month and day selection inputs
- Predicted visitor count display
- Crowd level classification (low / moderate / high)
- Visiting day recommendation
- Best alternative day suggestion within a 7-day window
- 7-day forecast visualization
- Detailed forecast result table
- Seasonal and Hajj-related contextual information where applicable

---

## How to Run Locally

**1. Clone the repository:**

```bash
git clone https://github.com/Nuha1abdul/intelligent-umrah-crowd-forecasting-system.git
cd intelligent-umrah-crowd-forecasting-system
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Launch the dashboard:**

```bash
streamlit run app.py
```

---

## Re-training the Model

To regenerate the trained model artifact from the processed dataset, execute:

```bash
python train_model.py
```

The script reads the following Excel file from the repository root:

```
Intelligent System for Predicting Visitor Crowd Levels and Optimal Visiting Times at Al-Masjid Al-Haram.xlsx
```

The training procedure uses Hijri year 1445H for training and Hijri year 1446H for evaluation. The output is a refreshed `growth_rate_xgboost_model.pkl`.

---

## Requirements

All Python dependencies are specified in `requirements.txt`. The core libraries used in this project are:

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `xgboost`
- `scikit-learn`
- `openpyxl`
- `joblib`

The deployment environment targets **Python 3.11**, as declared in `runtime.txt`.

---

## Deployment

The dashboard is deployed via **Streamlit Community Cloud**, connected directly to this GitHub repository. Any update pushed to the main branch is automatically reflected in the deployed application.

The following files are required for successful deployment:

- `app.py` — main dashboard application
- `requirements.txt` — Python package dependencies
- `runtime.txt` — Python runtime version
- `growth_rate_xgboost_model.pkl` — trained model artifact
- `growth_rate_xgboost_results_1446.xlsx` — precomputed forecast results
- Processed dataset file (repository root)

---

## Project Deliverables

This repository contains the technical implementation of the Intelligent Umrah Crowd Forecasting System, including the modeling workflow, deployment code, datasets, trained model artifact, and live dashboard.

| Component | Description |
|---|---|
| Modeling notebook | Complete data science workflow in `notebooks/final_model_notebook.ipynb` |
| Training script | Reproducible model training pipeline in `train_model.py` |
| Dashboard application | Streamlit interface implemented in `app.py` |
| Processed dataset | Final modeling dataset stored in `data/processed/` and repository root |
| Raw datasets | Original source datasets stored in `data/raw/` |
| External calendar files | Hijri calendar files stored in `data/external/` |
| Trained model | Serialized Growth-Rate XGBoost model in `growth_rate_xgboost_model.pkl` |
| Forecast results | Hijri year 1446 forecast outputs in `growth_rate_xgboost_results_1446.xlsx` |
| Environment setup | Python dependencies in `requirements.txt` and runtime configuration in `runtime.txt` |
| Live deployment | Public Streamlit dashboard linked in the Live Dashboard section |

The repository is organized to support reproducibility, model inspection, and dashboard deployment. Official GP2 materials such as the final report, poster, and presentation slides are organized in the `reports/`, `poster/`, and `presentation/` folders.

---

## Academic Context

This project was developed as a Data Science Graduation Project 2 (GP2) at Umm Al-Qura University, College of Computing. The work demonstrates an end-to-end applied data science pipeline: data preparation, feature engineering, forecasting model development, evaluation, and deployment through an interactive dashboard.
---

## Authors

Data Science Graduation Project           
Umm Al-Qura University  
College of Computing — Data Science Department
