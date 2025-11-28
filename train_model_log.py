"""
Train an XGBoost regressor with log1p target transform, plant-level holdout,
and optional cross-validation. Saves model and artifacts; predictions are
reported back in the original scale (expm1).

Usage example:
  python train_model_log.py --data-path datos_solcast_full-1764069958506.csv --run-name run_log1p
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb


TARGET = "valor_teorico"
DROP_COLS = ["planta_id", "id", "fecha", "id_planta", "anio", "mes", "dia", "dia_semana"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost with log1p target.")
    parser.add_argument("--data-path", type=Path, required=True, help="CSV con los datos (debe tener columna fecha).")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Carpeta para guardar modelos.")
    parser.add_argument("--run-name", type=str, default=None, help="Nombre del run (default: timestamp).")
    parser.add_argument("--holdout-frac", type=float, default=0.20, help="Fracción mínima de filas en holdout por plantas.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Tamaño de test dentro de plantas vistas.")
    parser.add_argument("--cv-folds", type=int, default=6, help="Número de folds para cross-validation (0/1 para desactivar).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla.")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def add_time_features(df_src: pd.DataFrame) -> pd.DataFrame:
    df = df_src.sort_values("fecha").copy()
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["dia"] = df["fecha"].dt.day
    df["dia_semana"] = df["fecha"].dt.dayofweek
    return df


def select_holdout_plants(df: pd.DataFrame, target_frac: float, seed: int) -> Tuple[List[int], float]:
    counts = df["planta_id"].value_counts()
    plant_ids = counts.index.to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(plant_ids)

    holdout_plants: List[int] = []
    rows_acc = 0
    total_rows = len(df)

    for pid in plant_ids:
        holdout_plants.append(int(pid))
        rows_acc += int(counts[pid])
        if rows_acc >= target_frac * total_rows:
            break

    frac_rows = rows_acc / total_rows
    return holdout_plants, frac_rows


def build_xy(df_src: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df_time = add_time_features(df_src)
    mask = df_time[TARGET].notna() & np.isfinite(df_time[TARGET])
    dropped = len(df_time) - mask.sum()
    if dropped:
        print(f"Filas descartadas por target nulo/no finito: {dropped}")
    df_clean = df_time.loc[mask].reset_index(drop=True)
    if "potencia" in df_clean.columns:
        df_clean["potencia_kw"] = df_clean["potencia"] / 1000.0
    X = df_clean[[c for c in df_clean.columns if c not in DROP_COLS + [TARGET]]].copy()
    y = df_clean[TARGET].copy()
    return df_clean, X, y


def collect_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred, squared=True)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def fit_model(X_train: pd.DataFrame, y_log_train: np.ndarray, args: argparse.Namespace) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train, y_log_train, verbose=False)
    return model


def main() -> None:
    args = parse_args()
    start_time = time.time()

    df = pd.read_csv(args.data_path, parse_dates=["fecha"])
    print(f"Cargado: {len(df):,} filas, {df['planta_id'].nunique()} plantas")

    if (df[TARGET] <= -1).any():
        raise ValueError("La transformación log1p requiere TARGET > -1 en todas las filas.")

    holdout_plants, holdout_frac_rows = select_holdout_plants(df, args.holdout_frac, args.seed)
    df_holdout_raw = df[df["planta_id"].isin(holdout_plants)].copy()
    df_train_raw = df[~df["planta_id"].isin(holdout_plants)].copy()

    print(f"Holdout plantas (seed {args.seed}): {holdout_plants}")
    print(f"Filas holdout: {len(df_holdout_raw):,} ({holdout_frac_rows:.1%} del total)")
    print(f"Filas train: {len(df_train_raw):,} en {df_train_raw['planta_id'].nunique()} plantas")

    train_df, X_all, y_all = build_xy(df_train_raw)
    feature_cols = X_all.columns.tolist()
    median_values = X_all.median(numeric_only=True)
    X_all = X_all.fillna(median_values)

    holdout_df, X_holdout, y_holdout = build_xy(df_holdout_raw)
    X_holdout = X_holdout.reindex(columns=feature_cols).fillna(median_values)

    y_log_all = np.log1p(y_all.values)
    y_log_holdout = np.log1p(y_holdout.values)

    X_train, X_test, y_log_train, y_log_test, y_train_raw, y_test_raw = train_test_split(
        X_all, y_log_all, y_all.values, test_size=args.test_size, random_state=args.seed
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    model = fit_model(X_train, y_log_train, args)

    # Predictions back to original scale
    train_pred = np.expm1(model.predict(X_train))
    test_pred = np.expm1(model.predict(X_test))
    holdout_pred = np.expm1(model.predict(X_holdout))

    metrics = {
        "train": collect_metrics(y_train_raw, train_pred),
        "test": collect_metrics(y_test_raw, test_pred),
        "holdout": collect_metrics(y_holdout.values, holdout_pred),
    }

    cv_metrics = None
    if args.cv_folds and args.cv_folds > 1:
        print(f"Cross-validation ({args.cv_folds} folds) en plantas vistas...")
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_metrics_list = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all), 1):
            X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
            y_log_tr, y_log_val = y_log_all[tr_idx], y_log_all[val_idx]
            y_val_raw = y_all.values[val_idx]
            m = fit_model(X_tr, y_log_tr, args)
            val_pred = np.expm1(m.predict(X_val))
            cv_metrics_list.append(collect_metrics(y_val_raw, val_pred))
            print(f"  Fold {fold}: {cv_metrics_list[-1]}")
        cv_metrics = {
            k: float(np.mean([m[k] for m in cv_metrics_list]))
            for k in cv_metrics_list[0]
        }
        print("CV mean metrics:", json.dumps(cv_metrics, indent=2))

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.pkl")

    holdout_pred_df = holdout_df[["planta_id", "potencia", "fecha", TARGET]].copy()
    holdout_pred_df["pred"] = holdout_pred
    holdout_pred_df.to_csv(run_dir / "holdout_predictions.csv", index=False)

    metadata = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "data_path": str(args.data_path),
        "target_transform": "log1p",
        "holdout_plants": holdout_plants,
        "holdout_frac_rows": holdout_frac_rows,
        "train_rows": len(df_train_raw),
        "holdout_rows": len(df_holdout_raw),
        "feature_cols": feature_cols,
        "hyperparams": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "test_size": args.test_size,
            "cv_folds": args.cv_folds,
            "seed": args.seed,
        },
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "elapsed_seconds": time.time() - start_time,
    }

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artefactos guardados en: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
