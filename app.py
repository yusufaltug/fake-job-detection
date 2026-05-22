import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, REPORTS_DIR
from src.predict import predict_record

st.set_page_config(
    page_title="Sahte İş İlanı Tespit Sistemi",
    page_icon="🛡️",
    layout="wide",
)


def image_if_exists(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Grafik bulunamadı: {path.name}. Önce `python src/train_model.py --model all` çalıştır.")


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


st.markdown(
    """
    <style>
        :root {
            --app-bg: #0f1311;
            --surface: #171d1a;
            --surface-soft: #202823;
            --surface-raised: #222a25;
            --ink: #eef5f0;
            --muted: #a7b2ab;
            --line: rgba(238, 245, 240, 0.13);
            --primary: #54d0ad;
            --primary-dark: #33ad8c;
            --accent: #8fb7ff;
            --warning: #f0b86e;
            --shadow: rgba(0, 0, 0, 0.34);
        }

        .stApp {
            background:
                linear-gradient(140deg, rgba(84, 208, 173, 0.10) 0%, transparent 30rem),
                linear-gradient(180deg, #111714 0%, var(--app-bg) 52%, #0b0e0d 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(15, 19, 17, 0.88);
            backdrop-filter: blur(10px);
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #111613;
            border-right: 1px solid var(--line);
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stWidgetLabel"],
        label {
            color: var(--ink);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
            color: var(--muted);
        }

        .app-hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 18rem;
            gap: 1.6rem;
            align-items: center;
            margin-bottom: 1.15rem;
            padding: 1.45rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(32, 40, 35, 0.94), rgba(18, 24, 21, 0.96));
            box-shadow: 0 18px 40px var(--shadow);
        }

        .app-eyebrow {
            margin: 0 0 0.6rem;
            color: var(--primary);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .app-hero h1 {
            margin: 0;
            max-width: 15ch;
            color: var(--ink);
            font-size: 2.9rem;
            line-height: 1.06;
            letter-spacing: 0;
        }

        .app-hero p {
            max-width: 46rem;
            margin: 0.9rem 0 0;
            color: var(--muted);
            font-size: 1.03rem;
            line-height: 1.65;
        }

        .hero-badge {
            padding: 1rem;
            color: var(--ink);
            border: 1px solid rgba(84, 208, 173, 0.26);
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            background: rgba(84, 208, 173, 0.08);
        }

        .hero-badge span {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .hero-badge strong {
            display: block;
            margin-top: 0.45rem;
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.35;
            letter-spacing: 0;
        }

        .quick-stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0 0 1.3rem;
        }

        .quick-stat {
            min-height: 5.1rem;
            padding: 0.95rem 1rem;
            border: 1px solid var(--line);
            border-top: 3px solid var(--primary);
            border-radius: 8px;
            background: var(--surface);
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.20);
        }

        .quick-stat:nth-child(2) {
            border-top-color: var(--accent);
        }

        .quick-stat:nth-child(3) {
            border-top-color: var(--warning);
        }

        .quick-stat b {
            display: block;
            color: var(--ink);
            font-size: 0.98rem;
            line-height: 1.35;
            letter-spacing: 0;
        }

        .quick-stat small {
            display: block;
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.35;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        p, li, span {
            color: inherit;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.4rem;
            border-bottom: 1px solid var(--line);
        }

        [data-testid="stTabs"] [role="tab"] {
            min-height: 2.75rem;
            padding: 0 1rem;
            border-radius: 8px 8px 0 0;
            color: var(--muted);
            background: rgba(255, 255, 255, 0.02);
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            color: var(--primary);
            background: var(--surface);
            border: 1px solid var(--line);
            border-bottom-color: var(--surface);
            font-weight: 700;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {
            color: var(--ink);
            border: 1px solid rgba(238, 245, 240, 0.16);
            border-radius: 8px;
            background: #111714;
        }

        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #7f8b85;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {
            border-color: rgba(84, 208, 173, 0.72);
            box-shadow: 0 0 0 1px rgba(84, 208, 173, 0.26);
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            color: var(--ink);
            border-color: rgba(238, 245, 240, 0.16);
            border-radius: 8px;
            background: #111714;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
            color: var(--ink);
            fill: var(--ink);
        }

        div[data-testid="stCheckbox"] label {
            padding: 0.2rem 0;
            color: var(--ink);
        }

        .stButton > button {
            width: 100%;
            min-height: 3rem;
            color: var(--ink);
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface-raised);
            font-weight: 800;
            letter-spacing: 0;
            box-shadow: 0 12px 24px rgba(84, 208, 173, 0.18);
        }

        .stButton > button[kind="primary"] {
            color: #07110e;
            border-color: var(--primary-dark);
            background: linear-gradient(180deg, var(--primary), var(--primary-dark));
        }

        .stButton > button[kind="primary"]:hover {
            border-color: var(--primary);
            background: linear-gradient(180deg, #6be0c0, var(--primary-dark));
        }

        [data-testid="stMetric"] {
            min-height: 7rem;
            padding: 1rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface);
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 800;
        }

        [data-testid="stImage"] img {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border-color: var(--line);
            background: var(--surface-soft);
            color: var(--ink);
        }

        hr {
            margin: 1.4rem 0;
            border-color: var(--line);
        }

        @media (max-width: 760px) {
            [data-testid="stAppViewContainer"] > .main .block-container {
                padding-top: 1rem;
            }

            .app-hero,
            .quick-stats {
                grid-template-columns: 1fr;
            }

            .app-hero {
                padding: 1.25rem;
            }

            .app-hero h1 {
                max-width: 100%;
                font-size: 2.15rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Proje Bilgisi")
    st.markdown(
        """
        **Proje:** Sahte İş İlanı Tespit Sistemi  
        **Yöntem:** NLP + Makine Öğrenmesi  
        **Öznitelik çıkarımı:** TF-IDF  
        **Çıktılar:** Gerçek/Sahte tahmini, risk skoru, açıklama  
        **Veri seti:** Real / Fake Job Posting Prediction
        """
    )
    st.info("Veri grafikleri için `python src/train_model.py`, model karşılaştırması için `python src/train_model.py --model all` çalıştırabilirsin.")

st.markdown(
    """
    <section class="app-hero">
        <div>
            <p class="app-eyebrow">NLP + Makine Öğrenmesi</p>
            <h1>Sahte İş İlanı Tespit Sistemi</h1>
            <p>
                Girilen iş ilanını model skoru ve açıklanabilir risk kurallarıyla değerlendirir;
                analiz çıktıları, veri seti grafikleri ve sunum notlarını tek arayüzde toplar.
            </p>
        </div>
        <div class="hero-badge">
            <span>Karar Destek</span>
            <strong>Risk skoru, tahmin sonucu ve gerekçeler birlikte gösterilir.</strong>
        </div>
    </section>
    <section class="quick-stats" aria-label="Uygulama özeti">
        <div class="quick-stat">
            <b>TF-IDF tabanlı analiz</b>
            <small>İlan metnindeki önemli ifadeler modele taşınır.</small>
        </div>
        <div class="quick-stat">
            <b>Açıklanabilir skor</b>
            <small>Kural sinyalleri nihai risk yorumunu destekler.</small>
        </div>
        <div class="quick-stat">
            <b>Sunuma hazır raporlar</b>
            <small>Veri ve performans grafikleri aynı panelde incelenir.</small>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_data, tab_model, tab_notes = st.tabs([
    "🔍 İlan Analizi",
    "📊 Veri Seti Analizi",
    "📈 Model Performansı",
    "🎬 Sunum İçin Notlar",
])

with tab_predict:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("İlan Metinleri")
        title = st.text_input("İlan başlığı", value="Remote Data Entry Assistant")
        company_profile = st.text_area("Şirket profili", height=120)
        description = st.text_area(
            "İş tanımı",
            height=180,
            value="Work from home opportunity with immediate start. Flexible hours and weekly payment.",
        )
        requirements = st.text_area("Gereksinimler", height=120, value="No experience required. Basic computer skills.")
        benefits = st.text_area("Yan haklar", height=100)

    with col2:
        st.subheader("Ek Bilgiler")
        location = st.text_input("Lokasyon", value="US, NY, New York")
        department = st.text_input("Departman", value="")
        salary_range = st.text_input("Maaş aralığı", value="")
        employment_type = st.selectbox(
            "İstihdam türü",
            ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"],
        )
        required_experience = st.selectbox(
            "Deneyim",
            ["", "Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"],
        )
        required_education = st.text_input("Eğitim", value="")
        industry = st.text_input("Sektör", value="")
        function = st.text_input("Fonksiyon", value="")

        telecommuting = st.checkbox("Uzaktan çalışma", value=True)
        has_company_logo = st.checkbox("Şirket logosu var", value=False)
        has_questions = st.checkbox("Başvuru soruları var", value=False)

    record = {
        "title": title,
        "location": location,
        "department": department,
        "salary_range": salary_range,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "telecommuting": int(telecommuting),
        "has_company_logo": int(has_company_logo),
        "has_questions": int(has_questions),
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
    }

    if st.button("Tahmin Et", type="primary"):
        try:
            result = predict_record(record, MODEL_PATH)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        st.divider()
        st.subheader("Tahmin Sonucu")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tahmin", result["prediction_label"])
        m2.metric("Model Sahte Olasılığı", f"%{result['model_risk_score']:.2f}")
        m3.metric("Final Risk Skoru", f"%{result['final_risk_score']:.2f}")
        m4.metric("Risk Seviyesi", result["risk_label"])

        if result["prediction"] == 1:
            st.error("Model bu ilanı SAHTE olma riski yüksek olarak işaretledi.")
        else:
            st.success("Model bu ilanı GERÇEK olma ihtimali daha yüksek olarak değerlendirdi.")

        st.caption(
            f"Kullanılan model: {result['model_name']} | Karar eşiği: {result['threshold']:.2f}"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Kural Tabanlı Açıklamalar")
            if result["rule_reasons"]:
                for reason in result["rule_reasons"]:
                    st.write(f"- {reason}")
            else:
                st.write("Belirgin kural tabanlı risk sinyali bulunmadı.")

        with c2:
            st.subheader("Modelin Öne Çıkardığı İfadeler")
            if result["model_terms"]:
                st.write(", ".join(result["model_terms"]))
            else:
                st.write("Bu örnek için öne çıkan kelime/token bulunamadı.")

with tab_data:
    st.header("📊 Veri Seti Analizi")
    summary = read_json_if_exists(REPORTS_DIR / "dataset_summary.json")
    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplam ilan", f"{summary.get('row_count', 0):,}")
        c2.metric("Sütun sayısı", summary.get("column_count", 0))
        c3.metric("Duplicate satır", summary.get("duplicate_rows", 0))

    col_a, col_b = st.columns(2)
    with col_a:
        image_if_exists(REPORTS_DIR / "class_distribution.png", "Veri setindeki gerçek/sahte dağılımı")
        image_if_exists(REPORTS_DIR / "text_length_by_class.png", "Sınıflara göre ilan metni uzunluğu")
    with col_b:
        image_if_exists(REPORTS_DIR / "missing_values_top.png", "Eksik değer analizi")
        if (REPORTS_DIR / "missing_values.csv").exists():
            st.subheader("Eksik Değer Tablosu")
            st.dataframe(pd.read_csv(REPORTS_DIR / "missing_values.csv"), use_container_width=True)

    st.subheader("Sınıflara Göre Sık Geçen İfadeler")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(REPORTS_DIR / "top_terms_fake.png", "Sahte ilanlarda sık geçen ifadeler")
    with c2:
        image_if_exists(REPORTS_DIR / "top_terms_real.png", "Gerçek ilanlarda sık geçen ifadeler")

with tab_model:
    st.header("📈 Model Performansı")
    metrics = read_json_if_exists(REPORTS_DIR / "metrics.json")
    if metrics and "best_model" in metrics:
        best = metrics["best_model"]
        st.subheader("En İyi Model Test Sonuçları")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Model", best.get("model_name", "-"))
        c2.metric("Accuracy", f"{best.get('accuracy', 0):.4f}")
        c3.metric("Precision", f"{best.get('precision', 0):.4f}")
        c4.metric("Recall", f"{best.get('recall', 0):.4f}")
        c5.metric("F1-score", f"{best.get('f1_score', 0):.4f}")
        st.caption(f"Karar eşiği validation verisiyle ayarlandı: {best.get('threshold', 0):.2f}")

    if (REPORTS_DIR / "model_comparison.csv").exists():
        st.subheader("Model Karşılaştırma Tablosu")
        st.dataframe(pd.read_csv(REPORTS_DIR / "model_comparison.csv"), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        image_if_exists(REPORTS_DIR / "model_comparison.png", "Accuracy, Precision, Recall, F1 ve ROC-AUC karşılaştırması")
        image_if_exists(REPORTS_DIR / "confusion_matrix.png", "Karmaşıklık matrisi")
        image_if_exists(REPORTS_DIR / "feature_importance.png", "Modelin kararında öne çıkan ifadeler")
    with col_b:
        image_if_exists(REPORTS_DIR / "roc_curve.png", "ROC eğrisi")
        image_if_exists(REPORTS_DIR / "precision_recall_curve.png", "Precision-Recall eğrisi")
        image_if_exists(REPORTS_DIR / "threshold_analysis.png", "Karar eşiği analizi")

with tab_notes:
    st.header("🎬 Sunumda Kullanabileceğin Kısa Anlatım")
    notes_path = REPORTS_DIR / "presentation_notes.md"
    if notes_path.exists():
        st.markdown(notes_path.read_text(encoding="utf-8"))
    else:
        st.warning("Sunum notları henüz oluşmadı. `python src/train_model.py --model all` komutunu çalıştır.")

    st.subheader("Video demo akışı")
    st.markdown(
        """
        1. Proje adını ve problemi tanıt.  
        2. Veri seti sekmesine geçip sınıf dengesizliğini ve eksik değerleri göster.  
        3. Model performansı sekmesine geçip F1-score, ROC-AUC, confusion matrix ve precision-recall grafiğini açıkla.  
        4. İlan analizi sekmesinde bir sahte ilan örneği dene.  
        5. Bir gerçek ilan örneği dene.  
        6. Sonuçta sistemin karar destek sistemi olduğunu, kesin hukuki karar vermediğini belirt.
        """
    )
