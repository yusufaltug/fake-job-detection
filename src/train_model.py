import argparse
import json
import sys
import gc
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Komut hem `python src/train_model.py` hem de `python -m src.train_model` ile çalışsın.
ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import DATA_PATH, MODEL_PATH, REPORTS_DIR, TARGET_COL, TEXT_COLS
from src.data_utils import load_dataset, prepare_training_frame


RANDOM_STATE = 42


def save_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def tune_threshold(y_true, probabilities) -> Dict[str, float]:
    """F1 skorunu en iyi yapan karar eşiğini validation verisi üzerinde bulur."""
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(round(threshold, 2)),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "f1": float(f1),
            }
    return best


def evaluate_with_threshold(
    name: str,
    pipeline: Pipeline,
    x_values,
    y_values,
    threshold: float,
    split_name: str,
) -> Dict[str, Any]:
    probabilities = pipeline.predict_proba(x_values)[:, 1]
    preds = (probabilities >= threshold).astype(int)
    return {
        "model_name": name,
        "split": split_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_values, preds)),
        "precision": float(precision_score(y_values, preds, zero_division=0)),
        "recall": float(recall_score(y_values, preds, zero_division=0)),
        "f1_score": float(f1_score(y_values, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_values, probabilities)),
        "average_precision": float(average_precision_score(y_values, probabilities)),
        "confusion_matrix": confusion_matrix(y_values, preds).tolist(),
        "classification_report": classification_report(y_values, preds, zero_division=0),
    }


def save_confusion_matrix(cm, labels: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_roc_curve(y_true, probabilities, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc_value = roc_auc_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Rastgele tahmin")
    ax.set_title("ROC Eğrisi")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_precision_recall_curve(y_true, probabilities, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    ap_value = average_precision_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, label=f"Average Precision = {ap_value:.4f}")
    ax.set_title("Precision-Recall Eğrisi")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_threshold_curve(y_true, probabilities, output_path: Path) -> None:
    rows = []
    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1_score": f1_score(y_true, preds, zero_division=0),
        })
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["threshold"], curve["precision"], label="Precision")
    ax.plot(curve["threshold"], curve["recall"], label="Recall")
    ax.plot(curve["threshold"], curve["f1_score"], label="F1-score")
    ax.set_title("Karar Eşiği Analizi")
    ax.set_xlabel("Karar eşiği")
    ax.set_ylabel("Skor")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_model_comparison_plot(comparison: pd.DataFrame, output_path: Path) -> None:
    if comparison.empty:
        return
    plot_df = comparison.set_index("model_name")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Model Karşılaştırması")
    ax.set_xlabel("Model")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_feature_importance(pipeline: Pipeline, model_name: str, output_path: Path, csv_path: Path) -> List[Dict[str, Any]]:
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    if model_name == "logistic_regression":
        scores = model.coef_[0]
        order = np.argsort(scores)[-20:][::-1]
        title = "Sahte İlan Kararını Güçlendiren İlk 20 İfade"
        label = "Katsayı"
    elif model_name == "complement_nb":
        scores = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        order = np.argsort(scores)[-20:][::-1]
        title = "Sahte İlan Sınıfında Öne Çıkan İlk 20 İfade"
        label = "Log olasılık farkı"
    elif model_name == "random_forest":
        scores = model.feature_importances_
        order = np.argsort(scores)[-20:][::-1]
        title = "Random Forest Öznitelik Önemleri"
        label = "Önem skoru"
    else:
        return []

    terms = feature_names[order]
    values = scores[order]
    feature_df = pd.DataFrame({"term": terms, "score": values})
    feature_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_df["term"][::-1], feature_df["score"][::-1])
    ax.set_title(title)
    ax.set_xlabel(label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return feature_df.to_dict(orient="records")


def to_dense(matrix):
    """Random Forest yoğun matris ister; TF-IDF sparse çıktısını dense yapar."""
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def build_pipeline(model_name: str) -> Pipeline:
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
            min_df=5,
            max_df=0.95,
            ngram_range=(1, 1),
            sublinear_tf=True,
        )
    elif model_name == "complement_nb":
        classifier = ComplementNB(alpha=0.5)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
            min_df=5,
            max_df=0.95,
            ngram_range=(1, 1),
            sublinear_tf=True,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=25,
            random_state=RANDOM_STATE,
            n_jobs=1,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
            max_depth=16,
            max_features="sqrt",
        )
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=350,
            min_df=8,
            max_df=0.95,
            ngram_range=(1, 1),
            sublinear_tf=True,
        )
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    steps = [("tfidf", vectorizer)]
    if model_name == "random_forest":
        steps.append(("to_dense", FunctionTransformer(to_dense, accept_sparse=True)))
    steps.append(("model", classifier))
    return Pipeline(steps)


def combined_text_for_eda(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    for col in TEXT_COLS:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    return work[TEXT_COLS].agg(" ".join, axis=1)


def save_dataset_graphs(raw_df: pd.DataFrame, reports_dir: Path) -> Dict[str, Any]:
    summary = {
        "row_count": int(len(raw_df)),
        "column_count": int(raw_df.shape[1]),
        "target_counts": raw_df[TARGET_COL].value_counts(dropna=False).to_dict() if TARGET_COL in raw_df.columns else {},
        "missing_values": raw_df.isna().sum().sort_values(ascending=False).to_dict(),
        "duplicate_rows": int(raw_df.duplicated().sum()),
    }
    save_json(summary, reports_dir / "dataset_summary.json")
    pd.Series(summary["missing_values"]).rename_axis("column").reset_index(name="missing_count").to_csv(
        reports_dir / "missing_values.csv", index=False
    )

    # 1) Sınıf dağılımı
    class_counts = raw_df[TARGET_COL].map({0: "Gerçek", 1: "Sahte"}).value_counts().reindex(["Gerçek", "Sahte"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(class_counts.index, class_counts.values)
    ax.set_title("Veri Seti Sınıf Dağılımı")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("İlan sayısı")
    for i, value in enumerate(class_counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(reports_dir / "class_distribution.png", dpi=180)
    plt.close(fig)

    # 2) Eksik değerler
    missing = raw_df.isna().sum().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(missing.index[::-1], missing.values[::-1])
    ax.set_title("En Çok Eksik Değer İçeren Sütunlar")
    ax.set_xlabel("Eksik değer sayısı")
    fig.tight_layout()
    fig.savefig(reports_dir / "missing_values_top.png", dpi=180)
    plt.close(fig)

    # 3) Metin uzunluğu dağılımı
    text = combined_text_for_eda(raw_df)
    length_df = pd.DataFrame({
        "text_length": text.str.len(),
        "class": raw_df[TARGET_COL].map({0: "Gerçek", 1: "Sahte"}),
    })
    data_to_plot = [
        length_df.loc[length_df["class"] == "Gerçek", "text_length"],
        length_df.loc[length_df["class"] == "Sahte", "text_length"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot(data_to_plot, tick_labels=["Gerçek", "Sahte"], showfliers=False)
    ax.set_title("Metin Uzunluğu Karşılaştırması")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("Karakter sayısı")
    fig.tight_layout()
    fig.savefig(reports_dir / "text_length_by_class.png", dpi=180)
    plt.close(fig)

    # 4) Gerçek ve sahte sınıflarda en sık geçen terimler
    # CountVectorizer yerine hafif bir Counter kullanıyoruz; böylece kullanıcı bilgisayarında daha hızlı çalışır.
    token_pattern = re.compile(r"[a-zA-Z]{3,}")

    def top_terms_for_class(class_value: int, top_n: int = 20) -> pd.DataFrame:
        counter = Counter()
        class_texts = text[raw_df[TARGET_COL].astype(int).values == class_value]
        # Gerçek sınıf çok büyük olduğu için hızlı grafik üretimi adına örneklem alıyoruz.
        # Sahte sınıf az olduğu için tamamı kullanılır.
        if len(class_texts) > 2500:
            class_texts = class_texts.sample(2500, random_state=RANDOM_STATE)
        for document in class_texts:
            tokens = [t.lower() for t in token_pattern.findall(document)]
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
            counter.update(tokens)
        return pd.DataFrame(counter.most_common(top_n), columns=["term", "count"])

    for class_value, file_name, title in [
        (1, "top_terms_fake.png", "Sahte İlanlarda En Sık Geçen İfadeler"),
        (0, "top_terms_real.png", "Gerçek İlanlarda En Sık Geçen İfadeler"),
    ]:
        top_terms = top_terms_for_class(class_value)
        top_terms.to_csv(reports_dir / file_name.replace(".png", ".csv"), index=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_terms["term"][::-1], top_terms["count"][::-1])
        ax.set_title(title)
        ax.set_xlabel("Geçme sayısı")
        fig.tight_layout()
        fig.savefig(reports_dir / file_name, dpi=180)
        plt.close(fig)

    del text
    gc.collect()
    return summary


def split_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """%70 train, %15 validation, %15 test olacak şekilde stratified bölme yapar."""
    x = df["model_text"]
    y = df[TARGET_COL].astype(int)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    validation_ratio = 0.15 / 0.85
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_ratio,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(description="Sahte iş ilanı tespit modeli eğitimi")
    parser.add_argument("--data", default=str(DATA_PATH), help="CSV veri seti yolu")
    parser.add_argument("--model-out", default=str(MODEL_PATH), help="Kaydedilecek model yolu")
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR), help="Rapor çıktıları klasörü")
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "complement_nb", "random_forest", "all"],
        default="logistic_regression",
        help="Eğitilecek model. Sunum için `--model all` önerilir.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(args.data)
    df = prepare_training_frame(raw_df)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(df)

    model_names = ["logistic_regression", "random_forest"] if args.model == "all" else [args.model]
    results = []
    fitted = {}
    validation_scores = {}

    for name in model_names:
        print(f"\nEğitiliyor: {name}", flush=True)
        pipeline = build_pipeline(name)
        pipeline.fit(x_train, y_train)

        val_probabilities = pipeline.predict_proba(x_val)[:, 1]
        threshold_info = tune_threshold(y_val, val_probabilities)
        validation_scores[name] = threshold_info

        test_metrics = evaluate_with_threshold(
            name=name,
            pipeline=pipeline,
            x_values=x_test,
            y_values=y_test,
            threshold=threshold_info["threshold"],
            split_name="test",
        )
        test_metrics["validation_threshold_tuning"] = threshold_info
        results.append(test_metrics)
        fitted[name] = pipeline

        print(
            f"{name}: threshold={threshold_info['threshold']:.2f}, "
            f"F1={test_metrics['f1_score']:.4f}, "
            f"Precision={test_metrics['precision']:.4f}, "
            f"Recall={test_metrics['recall']:.4f}, "
            f"ROC-AUC={test_metrics['roc_auc']:.4f}"
        )

    comparison = pd.DataFrame([
        {
            "model_name": r["model_name"],
            "threshold": r["threshold"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"],
            "roc_auc": r["roc_auc"],
            "average_precision": r["average_precision"],
        }
        for r in results
    ])
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    print("Model karşılaştırma tablosu kaydediliyor...", flush=True)
    save_model_comparison_plot(comparison, reports_dir / "model_comparison.png")
    print("Model karşılaştırma grafiği kaydedildi.", flush=True)

    # En iyi modeli validation F1'e göre seçiyoruz. Böylece test verisine göre model seçme hatası azaltılır.
    print("En iyi model seçiliyor...", flush=True)
    best_name = max(model_names, key=lambda n: validation_scores[n]["f1"])
    best_pipeline = fitted[best_name]
    best_result = next(r for r in results if r["model_name"] == best_name)

    print(f"En iyi model: {best_name}. Performans grafikleri oluşturuluyor...", flush=True)
    best_probabilities = best_pipeline.predict_proba(x_test)[:, 1]
    save_confusion_matrix(best_result["confusion_matrix"], labels=["Gerçek", "Sahte"], output_path=reports_dir / "confusion_matrix.png")
    save_roc_curve(y_test, best_probabilities, reports_dir / "roc_curve.png")
    save_precision_recall_curve(y_test, best_probabilities, reports_dir / "precision_recall_curve.png")
    save_threshold_curve(y_val, best_pipeline.predict_proba(x_val)[:, 1], reports_dir / "threshold_analysis.png")
    feature_terms = save_feature_importance(
        best_pipeline,
        best_name,
        reports_dir / "feature_importance.png",
        reports_dir / "feature_importance.csv",
    )

    print("Model dosyası kaydediliyor...", flush=True)
    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "threshold": best_result["threshold"],
        "metrics": best_result,
        "feature_terms": feature_terms,
        "text": "Sahte iş ilanı tespiti için TF-IDF + makine öğrenmesi modeli.",
    }
    joblib.dump(bundle, args.model_out)

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"all_models": results, "best_model": best_result}, f, ensure_ascii=False, indent=2)

    with open(reports_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(best_result["classification_report"])

    # En iyi model dışındaki ağır modelleri bellekten temizliyoruz.
    # Özellikle Random Forest sonrası EDA grafiklerini üretirken Windows bilgisayarlarda yavaşlamayı önler.
    for model_key in list(fitted.keys()):
        if model_key != best_name:
            del fitted[model_key]
    try:
        del pipeline
    except NameError:
        pass
    gc.collect()

    if args.model != "all":
        print("Veri seti grafikleri oluşturuluyor...", flush=True)
        # Veri seti grafiklerini en sonda oluşturuyoruz.
        # Böylece büyük EDA matrisleri model eğitimi sırasında belleği yavaşlatmaz.
        save_dataset_graphs(raw_df, reports_dir)
    else:
        print("Model karşılaştırması tamamlandı. EDA grafikleri mevcut rapor klasöründe korunuyor.", flush=True)

    notes = f"""# Proje Sunumunda Kullanılabilecek Sonuç Notları

## Veri seti
- Toplam ilan sayısı: {len(raw_df):,}
- Gerçek ilan sayısı: {int((raw_df[TARGET_COL] == 0).sum()):,}
- Sahte ilan sayısı: {int((raw_df[TARGET_COL] == 1).sum()):,}
- Problem ciddi sınıf dengesizliği içerir. Bu yüzden sadece accuracy değil; precision, recall, F1-score ve ROC-AUC incelenmiştir.

## Eğitim yöntemi
- Metin alanları birleştirildi: title, company_profile, description, requirements, benefits.
- Kategorik ve ikili alanlar modele metinsel token olarak eklendi.
- Metinler TF-IDF ile sayısallaştırıldı.
- Veri %70 eğitim, %15 validation, %15 test olacak şekilde ayrıldı.
- Karar eşiği test verisinde değil, validation verisi üzerinde F1-score'a göre ayarlandı.

## En iyi model
- Model: {best_name}
- Karar eşiği: {best_result['threshold']:.2f}
- Accuracy: {best_result['accuracy']:.4f}
- Precision: {best_result['precision']:.4f}
- Recall: {best_result['recall']:.4f}
- F1-score: {best_result['f1_score']:.4f}
- ROC-AUC: {best_result['roc_auc']:.4f}

## Videoda gösterilecek grafikler
1. class_distribution.png: Veri setindeki gerçek/sahte ilan dengesizliği.
2. missing_values_top.png: En çok eksik değer içeren sütunlar.
3. text_length_by_class.png: Gerçek ve sahte ilanların metin uzunluğu karşılaştırması.
4. top_terms_fake.png ve top_terms_real.png: Sınıflara göre sık geçen kelimeler.
5. model_comparison.png: Modellerin metrik karşılaştırması.
6. confusion_matrix.png: Doğru/yanlış tahminlerin sayısal özeti.
7. roc_curve.png: Modelin sınıfları ayırma başarısı.
8. precision_recall_curve.png: Dengesiz veri seti için kritik performans grafiği.
9. threshold_analysis.png: Karar eşiği değiştikçe precision/recall/F1 değişimi.
10. feature_importance.png: Modelin sahte ilan kararında öne çıkardığı ifadeler.
"""
    with open(reports_dir / "presentation_notes.md", "w", encoding="utf-8") as f:
        f.write(notes)

    print(f"\nEn iyi model: {best_name}")
    print(f"Model kaydedildi: {args.model_out}")
    print(f"Raporlar ve grafikler kaydedildi: {reports_dir}")
    print("Sunum notları: reports/presentation_notes.md")


if __name__ == "__main__":
    main()
