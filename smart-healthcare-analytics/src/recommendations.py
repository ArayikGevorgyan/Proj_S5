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


def generate_vitals_recommendations(vitals: dict) -> list[str]:
    """
    Provide simple recommendations based on raw daily vital signs
    such as blood pressure and heart rate.
    """
    recs: list[str] = []

    try:
        sys_bp = float(vitals.get("SystolicBP", 0) or 0)
    except (TypeError, ValueError):
        sys_bp = 0.0
    try:
        dia_bp = float(vitals.get("DiastolicBP", 0) or 0)
    except (TypeError, ValueError):
        dia_bp = 0.0
    try:
        hr = float(vitals.get("HeartRate", 0) or 0)
    except (TypeError, ValueError):
        hr = 0.0

    if sys_bp and dia_bp:
        if sys_bp >= 140 or dia_bp >= 90:
            recs.append(
                "Blood pressure is in a high range today — avoid excess salt, reduce stress, "
                "and consider contacting your doctor if readings stay high."
            )
        elif sys_bp >= 130 or dia_bp >= 80:
            recs.append(
                "Blood pressure is slightly elevated — monitor closely this week and "
                "review diet, exercise, and sleep habits."
            )
        elif sys_bp and dia_bp:
            recs.append("Blood pressure is within the typical range — keep your current routine and continue monitoring regularly.")

    if hr:
        if hr > 100:
            recs.append(
                "Resting heart rate is high — ensure you are resting during measurement and "
                "talk to a healthcare professional if this persists."
            )
        elif hr < 50:
            recs.append(
                "Resting heart rate is quite low — if you are not an endurance athlete, "
                "consult your doctor about this reading."
            )

    if not recs:
        recs.append("Vitals entered. Continue tracking daily to identify trends over time.")

    seen: set[str] = set()
    unique: list[str] = []
    for r in recs:
        if r not in seen:
            unique.append(r)
            seen.add(r)
    return unique
