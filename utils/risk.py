def classify_risk_level(confidence: float) -> str:
    if confidence >= 80:
        return "baixo"
    elif confidence >= 60:
        return "medio"
    else:
        return "alto"
