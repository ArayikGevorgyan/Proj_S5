"""
Database helper utilities for managing patients and prediction logs via SQLAlchemy.
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

import pandas as pd
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .database import Patient, PredictionLog, session_scope


def _patient_to_dict(patient: Patient) -> dict:
    return {
        "PatientID": patient.patient_id,
        "Name": patient.name,
        "Age": patient.age,
        "Gender": patient.gender,
        "BMI": patient.bmi,
        "Glucose": patient.glucose,
        "BloodPressure": patient.blood_pressure,
        "Cholesterol": patient.cholesterol,
        "CreatedAt": patient.created_at,
    }


def add_patient(data: dict) -> str:
    """Insert a new patient record and return the app-level patient identifier."""
    with session_scope() as session:
        patient = Patient(
            patient_id=data.get("PatientID") or str(uuid.uuid4())[:8],
            name=data.get("Name", "Unknown").strip() or "Unknown",
            age=data.get("Age"),
            gender=data.get("Gender"),
            bmi=data.get("BMI"),
            glucose=data.get("Glucose"),
            blood_pressure=data.get("BloodPressure"),
            cholesterol=data.get("Cholesterol"),
        )
        session.add(patient)
        session.flush()
        return patient.patient_id


def get_all_patients() -> pd.DataFrame:
    with session_scope() as session:
        patients = session.execute(select(Patient)).scalars().all()
        data = [_patient_to_dict(p) for p in patients]
    return pd.DataFrame(data)


def search_patients(query: str) -> pd.DataFrame:
    if not query:
        return get_all_patients()

    search = f"%{query.strip().lower()}%"
    with session_scope() as session:
        stmt = select(Patient).where(
            or_(
                Patient.name.ilike(search),
                Patient.patient_id.ilike(search),
            )
        )
        patients = session.execute(stmt).scalars().all()
        data = [_patient_to_dict(p) for p in patients]
    return pd.DataFrame(data)


def _find_patient_by_public_id(session: Session, patient_id: str) -> Optional[Patient]:
    if not patient_id:
        return None
    stmt = select(Patient).where(Patient.patient_id == patient_id)
    return session.execute(stmt).scalar_one_or_none()


def save_prediction_log(
    patient_id: str,
    disease_type: str,
    prediction: dict,
    input_data: Optional[dict] = None,
    created_by: Optional[str] = None,
) -> None:
    """Persist a prediction record for analytics and auditing."""
    if not patient_id or not prediction:
        return

    with session_scope() as session:
        patient = _find_patient_by_public_id(session, patient_id)
        if not patient:
            return

        log_entry = PredictionLog(
            patient=patient,
            disease_type=(disease_type or "general").lower(),
            risk_score=prediction.get("risk_score", 0.0),
            result=prediction.get("result", "Unknown"),
            timestamp=pd.to_datetime(prediction.get("timestamp")).to_pydatetime()
            if prediction.get("timestamp")
            else None,
            input_data=json.dumps(input_data or prediction.get("input_data", {})),
            created_by=created_by,
        )
        session.add(log_entry)


# Backwards compatibility alias
log_prediction = save_prediction_log


def get_prediction_history(patient_id: Optional[str] = None) -> pd.DataFrame:
    """Return historical predictions, optionally filtered by patient."""
    with session_scope() as session:
        stmt = select(PredictionLog)
        if patient_id:
            patient = _find_patient_by_public_id(session, patient_id)
            if not patient:
                return pd.DataFrame()
            stmt = stmt.where(PredictionLog.patient_id == patient.id)

        logs = session.execute(stmt.order_by(PredictionLog.timestamp.asc())).scalars().all()
        data = []
        for log in logs:
            data.append(
                {
                    "PatientID": log.patient.patient_id if log.patient else None,
                    "DiseaseType": log.disease_type,
                    "RiskScore": log.risk_score,
                    "Result": log.result,
                    "Timestamp": log.timestamp,
                    "InputData": log.input_data,
                    "CreatedBy": log.created_by,
                }
            )
    return pd.DataFrame(data)


def seed_patients_from_csv(session: Session, csv_path: str) -> None:
    """Utility to import patients from legacy CSV into the database."""
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        patient = Patient(
            patient_id=row.get("PatientID") or str(uuid.uuid4())[:8],
            name=row.get("Name", "Unknown"),
            age=row.get("Age"),
            gender=row.get("Gender"),
            bmi=row.get("BMI"),
            glucose=row.get("Glucose"),
            blood_pressure=row.get("BloodPressure"),
            cholesterol=row.get("Cholesterol"),
        )
        session.add(patient)
