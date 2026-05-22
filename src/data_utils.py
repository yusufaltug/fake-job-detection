import re
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.config import TEXT_COLS, CATEGORICAL_COLS, BINARY_COLS, TARGET_COL, ALL_INPUT_COLS


def load_dataset(csv_path: str) -> pd.DataFrame:
    """CSV veri setini okur."""
    return pd.read_csv(csv_path)


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _tokenize_metadata(value: Any) -> str:
    """Kategorik değerleri TF-IDF'e eklenebilir güvenli token formatına çevirir."""
    if pd.isna(value) or str(value).strip() == "":
        return "missing"
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_")
    return token if token else "missing"


def build_model_text(df: pd.DataFrame) -> pd.Series:
    """
    Metin alanlarını, kategorik bilgileri ve eksik değer bayraklarını tek bir metne dönüştürür.
    Bu yaklaşım web arayüzünde tek ilan tahmini yapmayı da kolaylaştırır.
    """
    work = df.copy()

    for col in TEXT_COLS + CATEGORICAL_COLS:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)

    model_text = work[TEXT_COLS].agg(" ".join, axis=1)

    for col in CATEGORICAL_COLS:
        clean = (
            work[col]
            .str.lower()
            .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
            .str.strip("_")
        )
        clean = clean.where(clean != "", "missing")
        model_text = model_text + " " + col + "_" + clean

    for col in ["salary_range", "company_profile", "requirements", "benefits"]:
        if col not in work.columns:
            present = pd.Series(False, index=work.index)
        else:
            present = work[col].fillna("").astype(str).str.strip() != ""
        model_text = model_text + " " + col + "_" + np.where(present, "present", "missing")

    for col in BINARY_COLS:
        if col not in work.columns:
            work[col] = 0
        values = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int).astype(str)
        model_text = model_text + " " + col + "_" + values

    return model_text


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Eğitim için model_text alanını ekler ve hedef değişkeni kontrol eder."""
    missing_cols = [c for c in [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik zorunlu sütunlar: {missing_cols}")

    work = df.copy()
    work["model_text"] = build_model_text(work)
    return work


def prepare_single_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Streamlit veya manuel tahmin için gelen tek ilan sözlüğünü standart forma getirir."""
    normalized = {col: record.get(col, "") for col in ALL_INPUT_COLS}
    for col in BINARY_COLS:
        try:
            normalized[col] = int(normalized.get(col, 0))
        except Exception:
            normalized[col] = 0
    one_row = pd.DataFrame([normalized])
    normalized["model_text"] = build_model_text(one_row).iloc[0]
    return normalized
