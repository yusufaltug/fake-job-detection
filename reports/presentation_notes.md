# Proje Sunumunda Kullanılabilecek Sonuç Notları

## Veri seti
- Toplam ilan sayısı: 17,880
- Gerçek ilan sayısı: 17,014
- Sahte ilan sayısı: 866
- Problem ciddi sınıf dengesizliği içerir. Bu yüzden sadece accuracy değil; precision, recall, F1-score ve ROC-AUC incelenmiştir.

## Eğitim yöntemi
- Metin alanları birleştirildi: title, company_profile, description, requirements, benefits.
- Kategorik ve ikili alanlar modele metinsel token olarak eklendi.
- Metinler TF-IDF ile sayısallaştırıldı.
- Veri %70 eğitim, %15 validation, %15 test olacak şekilde ayrıldı.
- Karar eşiği test verisinde değil, validation verisi üzerinde F1-score'a göre ayarlandı.

## En iyi model
- Model: logistic_regression
- Karar eşiği: 0.66
- Accuracy: 0.9836
- Precision: 0.8359
- Recall: 0.8231
- F1-score: 0.8295
- ROC-AUC: 0.9899

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
