from typing import Dict, Any, List, Tuple

SUSPICIOUS_KEYWORDS: List[Tuple[str, int, str]] = [
    ("wire transfer", 18, "Para transferi ifadesi dolandırıcılık riski oluşturabilir."),
    ("western union", 18, "Western Union gibi transfer kanalları sahte ilanlarda görülebilir."),
    ("send money", 20, "Adaydan para göndermesini istemek çok yüksek risklidir."),
    ("payment", 8, "Ödeme vurgusu iş ilanı bağlamında ek kontrol gerektirir."),
    ("bank account", 16, "Banka hesabı talebi kişisel veri/finansal risk oluşturabilir."),
    ("credit card", 18, "Kredi kartı bilgisi talebi yüksek risklidir."),
    ("social security", 18, "Kimlik/SSN benzeri hassas bilgi talebi risklidir."),
    ("passport", 12, "Pasaport bilgisi talebi ek dikkat gerektirir."),
    ("processing fee", 20, "Başvuru/işlem ücreti talebi sahte ilan belirtisi olabilir."),
    ("training fee", 20, "Eğitim ücreti talebi sahte ilan belirtisi olabilir."),
    ("investment", 12, "Yatırım/sermaye vurgusu iş ilanında şüpheli olabilir."),
    ("unlimited income", 10, "Gerçekçi olmayan kazanç vaadi risk sinyalidir."),
    ("no experience", 6, "Deneyimsiz ve yüksek kazanç vaadi birlikte risk yaratabilir."),
    ("immediate start", 5, "Acele başlatma vurgusu sosyal mühendislik riski taşıyabilir."),
    ("work from home", 5, "Evden çalışma tek başına kötü değildir; ancak başka sinyallerle risk artar."),
]


def rule_based_risk(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modelden bağımsız açıklanabilir risk kuralları.
    Bu kurallar tek başına karar vermez; model olasılığını açıklamaya destek olur.
    """
    score = 0
    reasons: List[str] = []

    combined = " ".join(
        str(record.get(col, "") or "")
        for col in ["title", "company_profile", "description", "requirements", "benefits"]
    ).lower()

    if not str(record.get("company_profile", "") or "").strip():
        score += 18
        reasons.append("Şirket profili boş bırakılmış.")

    if not str(record.get("salary_range", "") or "").strip():
        score += 8
        reasons.append("Maaş aralığı belirtilmemiş.")

    if not str(record.get("requirements", "") or "").strip():
        score += 10
        reasons.append("Gereksinimler bölümü boş veya çok zayıf.")

    if not str(record.get("benefits", "") or "").strip():
        score += 5
        reasons.append("Yan haklar bilgisi yok.")

    try:
        has_logo = int(record.get("has_company_logo", 0))
    except Exception:
        has_logo = 0
    if has_logo == 0:
        score += 14
        reasons.append("Şirket logosu bulunmuyor.")

    try:
        has_questions = int(record.get("has_questions", 0))
    except Exception:
        has_questions = 0
    if has_questions == 0:
        score += 6
        reasons.append("Başvuru soruları bulunmuyor.")

    if len(combined.strip()) < 700:
        score += 10
        reasons.append("İlan metni kısa; yeterli kurumsal/detail bilgi içermeyebilir.")

    for keyword, weight, explanation in SUSPICIOUS_KEYWORDS:
        if keyword in combined:
            score += weight
            reasons.append(f"Şüpheli ifade bulundu: '{keyword}'. {explanation}")

    score = min(score, 100)
    return {"rule_score": score, "reasons": reasons[:10]}


def risk_label(score: float) -> str:
    if score >= 70:
        return "Yüksek Risk"
    if score >= 40:
        return "Orta Risk"
    return "Düşük Risk"
