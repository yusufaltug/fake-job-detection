# Sahte İş İlanı Tespit Sistemi

Bu proje, Real / Fake Job Posting Prediction veri seti üzerinde NLP ve makine öğrenmesi kullanarak sahte iş ilanı tespiti yapar. Uygulama Streamlit ile web arayüzü olarak sunulur.

## Klasör Yapısı

```text
fake-job-detection/
├── app.py
├── requirements.txt
├── data/
│   └── fake_job_postings.csv
├── models/
│   └── best_model.joblib
├── reports/
│   ├── class_distribution.png
│   ├── missing_values_top.png
│   ├── text_length_by_class.png
│   ├── top_terms_fake.png
│   ├── top_terms_real.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── threshold_analysis.png
│   ├── feature_importance.png
│   └── presentation_notes.md
└── src/
    ├── config.py
    ├── data_utils.py
    ├── predict.py
    ├── risk_rules.py
    └── train_model.py
```

## Kurulum

```bash
python -m pip install -r requirements.txt
```

## Modeli Eğitme ve Grafikleri Oluşturma

Sadece varsayılan Logistic Regression modelini eğitmek için:

```bash
python src/train_model.py
```

Model karşılaştırması yapmak için:

```bash
python src/train_model.py --model all
```

`python src/train_model.py` veri analizi grafiklerini oluşturur. `python src/train_model.py --model all` ise Logistic Regression ve Random Forest modellerini karşılaştırır. Oluşturulan başlıca çıktılar:

- Veri seti sınıf dağılımı
- Eksik değer grafiği
- Metin uzunluğu karşılaştırması
- Gerçek/sahte sınıflarda sık geçen ifadeler
- Model karşılaştırma tablosu
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Threshold analysis
- Feature importance grafiği
- Sunum notları

## Web Uygulamasını Çalıştırma

```bash
python -m streamlit run app.py
```

Açılmazsa tarayıcıdan şu adrese gir:

```text
http://localhost:8501
```

## Kullanılan Yöntemler

- Metin alanları birleştirildi: title, company_profile, description, requirements, benefits.
- Kategorik alanlar modele token olarak eklendi.
- TF-IDF ile metinler sayısal forma dönüştürüldü.
- Logistic Regression, Complement Naive Bayes ve Random Forest modelleri karşılaştırılabilir hale getirildi.
- Sınıf dengesizliği nedeniyle accuracy yanında precision, recall, F1-score, ROC-AUC ve average precision hesaplandı.
- Karar eşiği test verisinde değil, validation verisinde F1-score'a göre ayarlandı.
- Web arayüzünde makine öğrenmesi skoru + kural tabanlı risk açıklamaları birlikte gösterildi.
