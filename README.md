# ML Pipelines for Solar Theoretical Generation (Prototype)

## Goal
We aim to predict the theoretical generation of solar plants using Solcast weather data, plant power, and temporal features. The goal is to bypass the theoretical model for plants with incomplete information, mitigating data gaps and enabling loss breakdowns (identify energy associated with each loss). This is a prototype to test multiple models and their metrics per plant.

## Data
- `data/theorical_data.csv`: main dataset (`planta_id`, `fecha`, weather features, `valor_teorico`, etc.).
- `data/planta.csv`: plant metadata (id, name, power, etc.).

## Training
- Model scripts: `train_model.py` (XGB), `train_model_log.py` (XGB log1p), `train_model_rf.py`, `train_model_catboost.py`, `train_model_lgbm.py`, `train_model_elastic.py`, `train_model_svm.py`, `train_model_keras.py`, `train_model_xgb_grid.py` (grid, top by MAPE).
- Unified runner: `train_runner_all.ipynb` with a section per model; adjust params and run. Defaults to `data/theorical_data.csv`.
- Artifacts: saved in `models/<run>/` with `model.pkl`/`.keras`, `metadata.json`, `holdout_predictions.csv` (predictions on holdout, real scale). XGB grid also saves `models/xgbgrid_summary_*.json` for all combos.
- Pipelines add `potencia_kw` (power/1000) and temporal features.

## Backend/Frontend
- Backend (`backend/server.py`): reads `planta.csv` from `data/`, serves `/models`, `/models/{id}/plants`, `/models/{id}/plant/{planta_id}`, `/models/{id}/plant/{planta_id}/metrics`, `/models/{id}/plants/metrics`. Compatible with current artifacts.
- Frontend (React in `frontend/`): model/plant selector, global and per-plant metrics, time series with range slider, error bars, per-plant metrics table, hyperparams and CV if available.

## Key Dependencies
- Python: `xgboost`, `scikit-learn`, `pandas`, `numpy`, `catboost`, `lightgbm`, `tensorflow` (for keras), `fastapi`, `uvicorn`.
- Node: Vite + React.