# CropYield Predictor — Agricultural Intelligence & Analytics Platform

This package follows the Code-A-Nova Task 5 brief: agricultural data cleaning, 15+ EDA charts, domain feature engineering, at least five regression models, 5-fold cross-validation, tuning, joblib model export, Streamlit dashboard, and final PDF report. fileciteturn2file0L21-L28

## Dataset note
The included `data/sample/crop_yield_demo.csv` is **synthetic demonstration data**. It is included so the code can run immediately. For the real internship submission, download a suitable dataset from the Kaggle source named in the brief and save it as `data/raw/crop_yield.csv`. The script `download_dataset.py` explains the placement.

## Setup (Windows / VS Code)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

## Train / compare models
```powershell
python train.py
```
Training evaluates Linear Regression, Ridge, Random Forest, Gradient Boosting, and Extra Trees using 5-fold cross-validation and reports MAE, RMSE, and R². The best CV model is fitted and saved as `models/cropyield_pipeline.joblib`. The code uses a scikit-learn Pipeline and ColumnTransformer so imputation, scaling, encoding, and feature engineering remain part of the model object.

## Deliverables
- Clean/demo CSV and data dictionary
- EDA notebook with 19 exported visualizations
- Feature engineering notebook and documented feature rationale
- Trained model artifact (`.joblib`)
- Five-model CV comparison table and metrics
- Feature-importance, residual, and error distribution charts
- Streamlit dashboard with EDA, prediction, and model-comparison tabs
- Final PDF report
- Tests and setup instructions

These align with the deliverables listed on page 3 of the project brief. fileciteturn2file0L129-L139

## Dashboard
`streamlit run app.py` opens the interactive dashboard. The prediction tab accepts crop, region, season, soil, irrigation, rainfall, temperature, humidity, NPK, fertilizer, and area values.

## Important interpretation
The sample data and current packaged model are synthetic demo materials. The metrics and chart patterns are **not official agricultural research findings** and should not be represented as Kaggle dataset results. Retrain with the chosen real dataset before final submission.

## Optional MLflow
To log experiments locally, install requirements and use `mlflow ui` in a separate terminal. The training script's CV comparison is saved as CSV; MLflow integration can be extended for each model run.
