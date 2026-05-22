# Mini-Project

## Overview
This project provides a simple web app for exploring a dataset, viewing insights, and running predictions from a mock ML model.

## Project Details
- Purpose: demonstrate a lightweight data exploration and prediction workflow.
- Data source: `fake_job_postings.csv` (used for dataset browsing and insights).
- Insights: summary metrics are computed from the dataset and shown in the UI.
- Prediction: a mock model in `utils/ml_logic.py` powers the prediction endpoint.
- Views: routes are organized under `views/` for about, dataset, insights, and prediction pages.

## Quick Start

### 1) Create and activate a virtual environment
```powershell
python -m venv venv
.
venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

### 3) (Optional) Initialize the mock model
```powershell
python setup_mock_model.py
```

### 4) Run the app
```powershell
python app.py
```

### 5) Open in your browser
Navigate to `http://127.0.0.1:5000`.

## Project Structure
```
app.py
fake_job_postings.csv
metrics.json
README.md
requirements.txt
setup_mock_model.py
style.css
utils/
	ml_logic.py
views/
	about.py
	dataset.py
	insights.py
	predict.py
```

## Notes
- If you see permission errors when activating the venv on Windows, run PowerShell as your user and execute:
	`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.
- The dataset file `fake_job_postings.csv` is required for the dataset and insights views.
