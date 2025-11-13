# src/recommendations.py

def generate_recommendations(input_data: dict, prediction: dict) -> list[str]:
    """
    Produce simple, rule-based health recommendations based on inputs and predicted risk.
    input_data: numeric-encoded fields (e.g., Gender 0/1, Smoking 0/1, PhysicalActivity 0/1/2)
    prediction: {"prediction": 0/1, "risk_score": float, "result": "High Risk"/"Low Risk"}
    """
    recs = []
    risk = prediction.get("risk_score", 0.0)

    age = float(input_data.get("Age", 0))
    bmi = float(input_data.get("BMI", 0))
    glucose = float(input_data.get("Glucose", 0))
    bp = float(input_data.get("BloodPressure", 0))
    chol = float(input_data.get("Cholesterol", 0))
    smoking = int(input_data.get("Smoking", 0))
    activity = int(input_data.get("PhysicalActivity", 1))

    # General risk-based recommendations
    if risk >= 0.6:
        recs.append("High model-estimated risk — recommend a follow-up consultation within 2–4 weeks.")
        recs.append("Consider additional lab tests and continuous monitoring of key metrics.")
    else:
        recs.append("Low model-estimated risk — keep monitoring and maintain healthy lifestyle.")

    # Specific factor hints (simple clinical thresholds, not medical advice)
    if bmi >= 30:
        recs.append("BMI suggests obesity — consider dietitian referral and increased physical activity.")
    elif bmi >= 25:
        recs.append("BMI is in overweight range — focus on gradual weight reduction.")

    if glucose >= 126:
        recs.append("Fasting glucose is elevated — consider diabetes screening and diet adjustment.")
    elif glucose >= 100:
        recs.append("Impaired fasting glucose range — monitor weekly and reduce simple sugars.")

    if bp >= 130:
        recs.append("Blood pressure is elevated — encourage sodium reduction and regular BP checks.")

    if chol >= 240:
        recs.append("High cholesterol — consider lipid profile evaluation and dietary fat moderation.")
    elif chol >= 200:
        recs.append("Borderline cholesterol — increase fiber and re-check in 3 months.")

    if smoking == 1:
        recs.append("Smoking detected — offer cessation support and counseling resources.")

    if activity == 0:
        recs.append("Low physical activity — aim for ≥150 minutes/week of moderate exercise.")
    elif activity == 1:
        recs.append("Moderate activity — consider adding 1–2 more active days per week.")

    # Age-based reminder
    if age >= 45:
        recs.append("Age ≥45 — consider annual comprehensive metabolic screening.")

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for r in recs:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq
