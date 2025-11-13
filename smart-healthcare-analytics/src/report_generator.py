# src/report_generator.py
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def _ensure_dirs():
    os.makedirs("docs/plots", exist_ok=True)
    os.makedirs("docs/reports", exist_ok=True)


def _make_risk_bar(risk_score: float) -> str:
    """
    Create a simple horizontal risk bar image and return its path.
    """
    _ensure_dirs()
    path = "docs/plots/risk_bar.png"
    plt.figure(figsize=(5, 0.6))
    plt.barh([0], [risk_score], height=0.4)
    plt.xlim(0, 1)
    plt.yticks([])
    plt.xlabel("Risk (0–1)")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def generate_patient_report(
    patient: dict,
    prediction: Optional[dict] = None,
    shap_path: Optional[str] = None,
    out_dir: str = "docs/reports"
) -> str:
    """
    Generate a PDF report for a patient.
    patient: dict with at least Name, Age, Gender, BMI, Glucose, BloodPressure, Cholesterol
    prediction: dict from predict_risk(...) with risk_score/result (optional but recommended)
    shap_path: path to SHAP image if available (optional)
    Returns: filepath to the generated PDF
    """
    _ensure_dirs()
    os.makedirs(out_dir, exist_ok=True)

    filename = f"patient_{patient.get('Name','unknown').replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(out_dir, filename)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    x_margin, y_margin = 2 * cm, 2 * cm
    y = height - y_margin

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, "Smart Healthcare Analytics — Patient Report")
    y -= 1.2 * cm

    # Patient Info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_margin, y, "Patient Information")
    y -= 0.5 * cm

    c.setFont("Helvetica", 11)
    info_lines = [
        f"Name: {patient.get('Name','')}",
        f"PatientID: {patient.get('PatientID','N/A')}",
        f"Age: {patient.get('Age','')}    Gender: {patient.get('Gender','')}",
        f"BMI: {patient.get('BMI','')}    Glucose: {patient.get('Glucose','')}",
        f"Blood Pressure: {patient.get('BloodPressure','')}    Cholesterol: {patient.get('Cholesterol','')}",
        f"Generated at: {datetime.utcnow().isoformat(timespec='seconds')} (UTC)"
    ]
    for line in info_lines:
        c.drawString(x_margin, y, line)
        y -= 0.5 * cm

    y -= 0.4 * cm
    c.setStrokeColor(colors.black)
    c.line(x_margin, y, width - x_margin, y)
    y -= 0.8 * cm

    # Prediction Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_margin, y, "Prediction Summary")
    y -= 0.6 * cm

    if prediction:
        result = prediction.get("result", "N/A")
        risk = prediction.get("risk_score", 0.0)
        c.setFont("Helvetica", 11)
        c.drawString(x_margin, y, f"Model Result: {result}  |  Risk Score: {risk:.2f}")
        y -= 0.8 * cm

        # Risk bar
        risk_img = _make_risk_bar(float(risk))
        c.drawImage(risk_img, x_margin, y - 1.5 * cm, width=8 * cm, height=1.2 * cm, preserveAspectRatio=True, mask='auto')
        y -= 2.0 * cm
    else:
        c.setFont("Helvetica", 11)
        c.drawString(x_margin, y, "No prediction provided.")
        y -= 0.8 * cm

    # SHAP plot (if available)
    if shap_path and os.path.exists(shap_path):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x_margin, y, "Explainable AI — SHAP Feature Importance")
        y -= 0.6 * cm
        c.drawImage(shap_path, x_margin, y - 8.5 * cm, width=16 * cm, height=8 * cm, preserveAspectRatio=True, mask='auto')
        y -= 9.0 * cm

    # Footer
    if y < 2 * cm:
        c.showPage()
        y = height - y_margin

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(x_margin, 1.2 * cm, "This report is generated for educational purposes and does not constitute medical advice.")
    c.save()
    return pdf_path
