from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np

from src.config import MODEL_PATH
from src.data_utils import prepare_single_record
from src.risk_rules import rule_based_risk, risk_label


def load_bundle(model_path: str | Path = MODEL_PATH) -> Dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}. Önce `python src/train_model.py` komutunu çalıştır."
        )
    return joblib.load(model_path)


def top_model_terms(model_text: str, pipeline, model_name: str, top_n: int = 8) -> List[str]:
    """Tek ilan için modelin dikkat ettiği başlıca kelime/token sinyallerini çıkarır."""
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    x_vec = vectorizer.transform([model_text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    if model_name == "logistic_regression" and hasattr(model, "coef_"):
        contributions = x_vec.multiply(model.coef_[0]).toarray()[0]
        positive_idx = np.where(contributions > 0)[0]
        if len(positive_idx) == 0:
            return []
        ordered = positive_idx[np.argsort(contributions[positive_idx])[::-1]][:top_n]
        return feature_names[ordered].tolist()

    if hasattr(model, "feature_importances_"):
        active = x_vec.toarray()[0] > 0
        values = model.feature_importances_ * active
        positive_idx = np.where(values > 0)[0]
        if len(positive_idx) == 0:
            return []
        ordered = positive_idx[np.argsort(values[positive_idx])[::-1]][:top_n]
        return feature_names[ordered].tolist()

    return []


def predict_record(record: Dict[str, Any], model_path: str | Path = MODEL_PATH) -> Dict[str, Any]:
    bundle = load_bundle(model_path)
    pipeline = bundle["pipeline"]
    model_name = bundle["model_name"]
    threshold = float(bundle.get("threshold", 0.5))

    normalized = prepare_single_record(record)
    model_text = normalized["model_text"]
    probability = float(pipeline.predict_proba([model_text])[0, 1])
    model_prediction = int(probability >= threshold)

    rules = rule_based_risk(normalized)
    final_score = round((probability * 100 * 0.75) + (rules["rule_score"] * 0.25), 2)

    return {
        "prediction": model_prediction,
        "prediction_label": "Sahte İlan" if model_prediction == 1 else "Gerçek İlan",
        "fake_probability": round(probability, 4),
        "model_risk_score": round(probability * 100, 2),
        "rule_score": rules["rule_score"],
        "final_risk_score": final_score,
        "risk_label": risk_label(final_score),
        "threshold": threshold,
        "model_name": model_name,
        "rule_reasons": rules["reasons"],
        "model_terms": top_model_terms(model_text, pipeline, model_name),
    }
