"""
Authentication and registration helpers backed by the SQLAlchemy database.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta

from sqlalchemy import select

try:
    import bcrypt
except ImportError:  # pragma: no cover
    bcrypt = None

from .database import session_scope, User, Patient, SessionToken
import uuid


@dataclass
class AuthenticatedUser:
    id: int
    username: str
    name: str
    role: str
    email: Optional[str] = None
    patient_db_id: Optional[int] = None
    patient_public_id: Optional[str] = None


def hash_password(password: str) -> str:
    if bcrypt:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    # Fallback (not recommended) – store plain text
    return password


def verify_password(password: str, hashed: str) -> bool:
    if bcrypt and hashed.startswith("$2"):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except ValueError:
            return False
    return password == hashed


def _user_to_dataclass(user: User) -> AuthenticatedUser:
    patient_public_id = user.patient.patient_id if user.patient else None
    patient_db_id = user.patient.id if user.patient else None
    return AuthenticatedUser(
        id=user.id,
        username=user.username,
        name=user.name,
        role=user.role,
        email=user.email,
        patient_db_id=patient_db_id,
        patient_public_id=patient_public_id,
    )


def authenticate_user(username: str, password: str) -> Optional[AuthenticatedUser]:
    with session_scope() as session:
        user = session.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if not user:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return _user_to_dataclass(user)


def _ensure_unique_user(session, username: str, email: Optional[str]):
    if session.execute(select(User).where(User.username == username)).first():
        raise ValueError("Username already exists. Please choose another.")
    if email:
        if session.execute(select(User).where(User.email == email)).first():
            raise ValueError("An account with this email already exists.")


def register_doctor_account(name: str, email: str, username: str, password: str) -> AuthenticatedUser:
    with session_scope() as session:
        _ensure_unique_user(session, username, email)
        user = User(
            username=username,
            name=name,
            email=email,
            role="doctor",
            password_hash=hash_password(password),
        )
        session.add(user)
        session.flush()
        return _user_to_dataclass(user)


def register_patient_account(
    name: str,
    email: str,
    username: str,
    password: str,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    bmi: Optional[float] = None,
    glucose: Optional[float] = None,
    blood_pressure: Optional[float] = None,
    cholesterol: Optional[float] = None,
) -> AuthenticatedUser:
    with session_scope() as session:
        _ensure_unique_user(session, username, email)
        patient = Patient(
            patient_id=str(uuid.uuid4())[:8],
            name=name,
            age=age,
            gender=gender,
            bmi=bmi,
            glucose=glucose,
            blood_pressure=blood_pressure,
            cholesterol=cholesterol,
        )
        session.add(patient)
        session.flush()

        user = User(
            username=username,
            name=name,
            email=email,
            role="patient",
            password_hash=hash_password(password),
            patient=patient,
        )
        session.add(user)
        session.flush()
        return _user_to_dataclass(user)


def get_patient_record_for_user(user: AuthenticatedUser) -> Optional[dict]:
    if not user.patient_db_id:
        return None
    with session_scope() as session:
        patient = session.get(Patient, user.patient_db_id)
        if not patient:
            return None
        return {
            "PatientID": patient.patient_id,
            "Name": patient.name,
            "Age": patient.age,
            "Gender": patient.gender,
            "BMI": patient.bmi,
            "Glucose": patient.glucose,
            "BloodPressure": patient.blood_pressure,
            "Cholesterol": patient.cholesterol,
        }


def create_session_token(user_id: int, ttl_hours: int = 24) -> str:
    token = uuid.uuid4().hex
    expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
    with session_scope() as session:
        session_token = SessionToken(token=token, user_id=user_id, expires_at=expires_at)
        session.add(session_token)
    return token


def get_user_by_token(token: str) -> Optional[AuthenticatedUser]:
    if not token:
        return None
    with session_scope() as session:
        session_token = session.get(SessionToken, token)
        if not session_token:
            return None
        if session_token.expires_at and session_token.expires_at < datetime.utcnow():
            session.delete(session_token)
            return None
        user = session_token.user
        if not user:
            session.delete(session_token)
            return None
        return _user_to_dataclass(user)


def invalidate_session_token(token: str) -> None:
    if not token:
        return
    with session_scope() as session:
        session_token = session.get(SessionToken, token)
        if session_token:
            session.delete(session_token)


def list_users_by_role(role: str) -> list[AuthenticatedUser]:
    """Return all users for a given role (doctor/patient)."""
    with session_scope() as session:
        users = session.execute(select(User).where(User.role == role)).scalars().all()
        return [_user_to_dataclass(u) for u in users]
