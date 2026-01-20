# forecasting/data/io.py
import pandas as pd
from pathlib import Path
import json
import torch
from pytorch_forecasting import TimeSeriesDataSet
from lightning.pytorch import Trainer


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


# TFT model helpers
def save_checkpoint(trainer: Trainer, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)


def load_checkpoint(model_cls, path: Path):
    return model_cls.load_from_checkpoint(path, weights_only=False)


# TimeSeriesDataSet helpers
def save_TimeSeriesDataSet(dataset: TimeSeriesDataSet, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(path)


def load_TimeSeriesDataSet(path: Path) -> TimeSeriesDataSet:
    # trusted local artifact
    return torch.load(path, weights_only=False)


# Dataset persistence helpers
def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
