"""
Train a simple Keras MLP with plant holdout and save artifacts compatible with the backend.
Predicciones y métricas se reportan en escala real (sin log).

Usage:
  python train_model_keras.py --data-path datos_solcast_full-1764069958506.csv --run-name keras_run
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
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


TARGET = "valor_teorico"
DROP_COLS = ["planta_id", "id", "fecha", "id_planta", "anio", "mes", "dia", "dia_semana"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Keras MLP with plant holdout.")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--holdout-frac", type=float, default=0.20)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-units", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    return p.parse_args()


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
    holdout, acc = [], 0
    total = len(df)
    for pid in plant_ids:
        holdout.append(int(pid))
        acc += int(counts[pid])
        if acc >= target_frac * total:
            break
    return holdout, acc / total


def build_xy(df_src: pd.DataFrame):
    df_time = add_time_features(df_src)
    mask = df_time[TARGET].notna() & np.isfinite(df_time[TARGET])
    if mask.sum() < len(df_time):
        print(f"Filas descartadas por target nulo/no finito: {len(df_time) - mask.sum()}")
    df_clean = df_time.loc[mask].reset_index(drop=True)
    X = df_clean[[c for c in df_clean.columns if c not in DROP_COLS + [TARGET]]].copy()
    y = df_clean[TARGET].copy()
    return df_clean, X, y


def collect_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred, squared=True)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def build_model(input_dim: int, args: argparse.Namespace) -> keras.Model:
    tf.keras.utils.set_random_seed(args.seed)
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in args.hidden_units:
        x = keras.layers.Dense(units, activation="relu")(x)
        if args.dropout > 0:
            x = keras.layers.Dropout(args.dropout)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mae", "mse"],
    )
    return model


def main():
    args = parse_args()
    start = time.time()

    df = pd.read_csv(args.data_path, parse_dates=["fecha"])
    print(f"Cargado: {len(df):,} filas, {df['planta_id'].nunique()} plantas")

    holdout_plants, holdout_frac = select_holdout_plants(df, args.holdout_frac, args.seed)
    df_holdout_raw = df[df["planta_id"].isin(holdout_plants)].copy()
    df_train_raw = df[~df["planta_id"].isin(holdout_plants)].copy()

    train_df, X_all, y_all = build_xy(df_train_raw)
    feature_cols = X_all.columns.tolist()
    med = X_all.median(numeric_only=True)
    X_all = X_all.fillna(med)

    holdout_df, X_holdout, y_holdout = build_xy(df_holdout_raw)
    X_holdout = X_holdout.reindex(columns=feature_cols).fillna(med)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.seed
    )

    model = build_model(X_train.shape[1], args)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    train_pred = model.predict(X_train, verbose=0).ravel()
    test_pred = model.predict(X_test, verbose=0).ravel()
    holdout_pred = model.predict(X_holdout, verbose=0).ravel()

    metrics = {
        "train": collect_metrics(y_train, train_pred),
        "test": collect_metrics(y_test, test_pred),
        "holdout": collect_metrics(y_holdout, holdout_pred),
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    run_name = args.run_name or f"keras_{len(args.hidden_units)}layers_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model.save(run_dir / "model.keras")
    joblib.dump(med, run_dir / "median_values.pkl")

    holdout_pred_df = holdout_df[["planta_id", "potencia", "fecha", TARGET]].copy()
    holdout_pred_df["pred"] = holdout_pred
    holdout_pred_df.to_csv(run_dir / "holdout_predictions.csv", index=False)

    metadata = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "data_path": str(args.data_path),
        "holdout_plants": holdout_plants,
        "holdout_frac_rows": holdout_frac,
        "train_rows": len(df_train_raw),
        "holdout_rows": len(df_holdout_raw),
        "feature_cols": feature_cols,
        "hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_units": args.hidden_units,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "test_size": args.test_size,
            "seed": args.seed,
        },
        "metrics": metrics,
        "target_transform": None,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artefactos guardados en: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
