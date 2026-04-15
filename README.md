# Fraud Detection ML System

Учебный end-to-end ML/backend проект для оценки вероятности мошеннической транзакции.  
Reproducible fraud detection project with a training pipeline, shared feature engineering for train and inference, FastAPI API, demo dashboard, and Docker setup for local launch.


## Overview

| Item | Details |
| --- | --- |
| Task | Binary classification for fraud detection |
| Dataset | `data/creditcard.csv` |
| Model | `RandomForestClassifier` |
| API | FastAPI (`/predict`, `/health`) |
| Dashboard | Static web UI at `/dashboard/` |
| Docker | `Dockerfile` + `docker-compose.yml` |

Final serving model: `RandomForestClassifier`, selected through staged model comparison with single-split baseline, stratified cross-validation, and separate threshold tuning.  
**Финальная serving-модель:** `RandomForestClassifier`, выбранная через поэтапное сравнение моделей.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- joblib
- FastAPI
- Pydantic
- Uvicorn
- Docker

## Project Components

## Project Components / Что есть в проекте

- Reproducible training pipeline from `data/creditcard.csv`  
  (воспроизводимый pipeline обучения)
- Shared feature engineering for training and inference  
  (единая логика признаков для train и inference)
- RandomForest serving model for fraud scoring  
  (основная модель для скоринга fraud)
- CatBoost baseline kept for model comparison  
  (CatBoost сохранён как сильный baseline-кандидат)
- Evaluation on a stratified test split  
- FastAPI service for prediction  
- Simple dashboard for manual inference checks  
- Docker setup for local demo launch

## Metrics

Current metrics from the main training pipeline on the test split:
**Ниже — основные метрики текущей serving-модели на test split.**

| Metric | Value |
| --- | ---: |
| PR-AUC | 0.876845 |
| ROC-AUC | 0.967647 |
| F2 | 0.788913 |
| Precision | 0.961039 |
| Recall | 0.755102 |
| Threshold | 0.50 |

These are the default evaluation metrics at `threshold = 0.5` from the main train/test pipeline.
The tuned operating threshold for the final `RandomForestClassifier` is reported separately in the **Model Selection** section and is currently `0.2`.

Why these metrics matter / Почему эти метрики важны:
- `PR-AUC` is more informative than accuracy for this dataset because fraud cases are rare.  
  (`PR-AUC` важнее accuracy из-за сильного дисбаланса классов.)
- `Recall` is important because missing fraudulent transactions is costly.  
  (`Recall` важен, потому что пропуск fraud-операции дорогой.)
- `F2` gives more weight to recall, which fits the fraud detection setting better than a symmetric metric.  
  (`F2` сильнее акцентирует recall.)

## Quick Start

Minimal local flow:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.main:app --reload
```

After startup:
- Dashboard: `http://localhost:8000/dashboard/`
- Health check: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`

## Problem Context

The project treats fraud detection as a binary classification task: for each transaction, the model predicts the probability that it belongs to the fraud class.

The dataset contains:
- `Time`
- `Amount`
- `V1` ... `V28`
- `Class` as the target

Why the task is non-trivial:
- the dataset is strongly imbalanced, with fraud transactions representing a very small minority;
- the API works with the transformed feature space from the dataset, not with raw business transaction fields;
- because of the imbalance, a high accuracy would not necessarily mean that the model is useful.

## ML Pipeline

Current pipeline:

1. Load data from `data/creditcard.csv`
2. Build the final feature set
3. Split data into train/test using stratification
4. Train a `RandomForestClassifier`
5. Evaluate on the test split
6. Save the trained model to `models/random_forest_fraud_model.joblib`
7. Load the saved model in FastAPI and use it in `/predict`

The serving artifact is a serialized sklearn `Pipeline`. It includes both the fitted `RandomForestClassifier` and the preprocessing needed for serving, including one-hot encoding for `time_bin`.

Current model parameters:
- `n_estimators=200`
- `class_weight='balanced'`
- `n_jobs=-1`
- `random_state=42`

## Model Selection / Как выбиралась инальная модель

This project uses a process-driven model selection approach.  
**Выбор модели здесь сделан как отдельный осознанный процесс, а не как разовое решение по одному запуску.**

Model selection in this project was done in stages rather than by intuition alone.

1. A **single-split benchmark** was used as a quick preliminary baseline on one stratified train/test split.
2. A **stratified cross-validation benchmark** became the main model selection procedure.
3. `mean PR-AUC` across CV folds was used as the primary selection metric because the fraud class is strongly imbalanced.
4. After shortlist selection, **threshold tuning** was performed separately on a validation split using `F2`.
5. Based on this process, `RandomForestClassifier` was chosen as the final serving model.

Model selection summary / Кратко:
- single-split benchmark is kept as a fast baseline comparison;
- cross-validation benchmark is the main source of model selection;
- `CatBoostClassifier` remains in the repository as a strong benchmark candidate, not as the final serving model.

- single-split benchmark сохранён как быстрый baseline;
- CV benchmark используется как основной этап выбора модели;
- `CatBoostClassifier` остался в проекте как сильный кандидат, но не как финальная serving-модель.

Cross-validation benchmark summary:

| model | mean PR-AUC | std PR-AUC | mean ROC-AUC | std ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| RandomForestClassifier | 0.855822 | 0.024644 | 0.954049 | 0.010747 |
| CatBoostClassifier | 0.811840 | 0.038346 | 0.981338 | 0.001592 |
| LogisticRegression | 0.760740 | 0.014065 | 0.977431 | 0.007064 |

Threshold tuning on shortlisted models:

| model | threshold | precision | recall | f2 |
| --- | ---: | ---: | ---: | ---: |
| RandomForestClassifier | 0.2 | 0.924731 | 0.877551 | 0.886598 |
| CatBoostClassifier | 0.9 | 0.833333 | 0.867347 | 0.860324 |

This threshold analysis is intentionally kept separate from model selection:
- model selection is based on cross-validation `mean PR-AUC`;
- operating threshold selection is based on `F2` for shortlisted models on a separate validation split;
- for the final serving model, the tuned operating threshold is `0.2`, while the headline training metrics above are still reported at the default `0.5`.

## Feature Engineering / Построение признаков

The model uses 33 features.

Base features:
- `Time`
- `Amount`
- `V1` ... `V28`

Derived features:
- `hour` — extracted from `Time` to capture time-of-day patterns
- `log_amount` — `log1p(Amount)` to reduce the effect of a long-tailed amount distribution
- `time_bin` — categorical time bucket: `night`, `morning`, `afternoon`, `evening`

Important detail / Важный момент:
- the same feature engineering logic is reused for both training and inference, so train and serve stay aligned.
- одна и та же логика признаков используется и при обучении, и при инференсе, поэтому нет train/inference mismatch.

## API

### `GET /`

Redirects to the dashboard:

- `http://localhost:8000/` -> `http://localhost:8000/dashboard/`

### `GET /health`

Returns service status:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Accepts one transaction in the dataset feature format and returns the fraud probability.

Example response:

```json
{
  "fraud_proba": 0.39
}
```

<details>
<summary>Example request payload</summary>

```json
{
  "Time": 0.0,
  "Amount": 149.62,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "V3": 2.53634673796914,
  "V4": 1.37815522427443,
  "V5": -0.338320769942518,
  "V6": 0.462387777762292,
  "V7": 0.239598554061257,
  "V8": 0.0986979012610507,
  "V9": 0.363786969611213,
  "V10": 0.0907941719789316,
  "V11": -0.551599533260813,
  "V12": -0.617800855762348,
  "V13": -0.991389847235408,
  "V14": -0.311169353699879,
  "V15": 1.46817697209427,
  "V16": -0.470400525259478,
  "V17": 0.207971241929242,
  "V18": 0.0257905801985591,
  "V19": 0.403992960255733,
  "V20": 0.251412098239705,
  "V21": -0.018306777944153,
  "V22": 0.277837575558899,
  "V23": -0.110473910188767,
  "V24": 0.0669280749146731,
  "V25": 0.128539358273528,
  "V26": -0.189114843888824,
  "V27": 0.133558376740387,
  "V28": -0.0210530534538215
}
```

</details>

Notes:
- the API expects the same feature schema as the dataset;
- this is useful for demonstrating the ML pipeline, but it is not a realistic public interface for a production antifraud service.

## Dashboard

The repository includes a simple web UI at `/dashboard/`.

It allows you to:
- enter transaction features manually;
- load a sample transaction;
- send a request to `/predict`;
- inspect the returned fraud probability in a lightweight interface.

This dashboard is intended for demo and manual verification, not for analytics or operations work.

## Project Structure

```text
fraud-detection-ml-system/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app/
│   ├── main.py
│   └── utils.py
├── dashboard/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── data/
│   └── creditcard.csv
├── models/
│   ├── catboost_fraud_model.cbm
│   └── random_forest_fraud_model.joblib
├── scripts/
│   ├── compare_models.py
│   ├── compare_models_cv.py
│   └── train_model.py
└── src/
    ├── features/
    │   └── fraud_features.py
    ├── inference/
    │   └── predict.py
    ├── models/
    │   └── model_utils.py
    ├── training/
    │   ├── compare_models.py
    │   ├── compare_models_cv.py
    │   └── train.py
    └── utils/
        └── metrics.py
```

Key files:
- `src/features/fraud_features.py` — shared feature engineering
- `src/training/train.py` — training, split, evaluation, model saving
- `src/training/compare_models.py` — single-split benchmark
- `src/training/compare_models_cv.py` — CV-based model selection and threshold tuning
- `scripts/train_model.py` — CLI entry point for model training
- `app/main.py` — FastAPI app and endpoints
- `models/random_forest_fraud_model.joblib` — main serving artifact as a serialized sklearn pipeline
- `models/catboost_fraud_model.cbm` — retained benchmark artifact

## Run Locally

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the dataset is available at `data/creditcard.csv`

4. Train the model:

```bash
python scripts/train_model.py
```

Optional arguments:

```bash
python scripts/train_model.py \
  --data-path data/creditcard.csv \
  --model-path models/random_forest_fraud_model.joblib \
  --test-size 0.2 \
  --random-state 42 \
  --threshold 0.5
```

5. Start the API:

```bash
uvicorn app.main:app --reload
```

## Run with Docker

Before using Docker, the model should already exist at `models/random_forest_fraud_model.joblib`, because the container starts the API with a ready model artifact.

Build and run:

```bash
docker compose up --build
```

After startup:
- `http://localhost:8000/dashboard/`
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

Current Docker limitation:
- this setup is intended for local demo usage and still copies both `models/` and `data/` into the container.

## Limitations / Ограничения

- this is a study/demo project, not a production deployment;
- the API expects PCA-style dataset features `V1` ... `V28`, not raw transaction fields;
- the main training metrics are still reported from a single stratified train/test split;
- the project does include a CV benchmark and threshold tuning, but not repeated CV, nested validation, or hyperparameter search;
- there is no experiment tracking;
- there are no automated tests for the training pipeline or API yet;
- there is no production logging, monitoring, or deployment setup;
- the dashboard is intentionally lightweight and not a full analytical tool;
- model selection is more robust than before, but it is still not backed by a full experiment-management workflow.

## Future Improvements

- threshold tuning based on business trade-offs
- tests for feature engineering, training, and `/predict`
- configuration/settings for model and pipeline parameters
- basic API logging and error reporting
- richer dashboard explanations
- saved metrics report and a short model card

## Skills Demonstrated / Что показывает проект 

- tabular ML for imbalanced classification  
- reproducible model training pipeline  
- process-driven model selection with baseline benchmark, CV benchmark, and threshold tuning  
- shared feature engineering for train and inference  
- FastAPI service for model inference  
- model packaging and local deployment with Docker  
- presenting an ML project end-to-end for portfolio and interview discussion  


- работа с табличными данными и сильным дисбалансом классов
- воспроизводимый pipeline обучения
- осознанный выбор модели через benchmark + CV + threshold tuning
- согласование train и inference
- ML-serving через FastAPI
- упаковка проекта в Docker
