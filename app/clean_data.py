from pathlib import Path
import re

import numpy as np
import pandas as pd

RAW_NULLS = {"", "na", "n/a", "null", "none", "missing", "unknown"}
KNOWN_NUMERIC_COLUMNS = {"tenure", "monthlycharges", "totalcharges", "seniorcitizen"}


def snake_case(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_text(value):
    if isinstance(value, str):
        cleaned = value.strip()
        return np.nan if cleaned.lower() in RAW_NULLS else cleaned
    return value


def clean_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df.columns = [snake_case(column) for column in df.columns]

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].map(normalize_text)

    for column in KNOWN_NUMERIC_COLUMNS.intersection(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
