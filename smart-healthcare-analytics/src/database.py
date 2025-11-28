import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_URL = f"sqlite:///{DATA_DIR / 'healthcare.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


# ---------------- MODELS ----------------
class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(36), unique=True, index=True, nullable=False)
    name = Column(String(120), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(32), nullable=True)
    bmi = Column(Float, nullable=True)
    glucose = Column(Float, nullable=True)
    blood_pressure = Column(Float, nullable=True)
    cholesterol = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("PredictionLog", back_populates="patient", cascade="all, delete-orphan")
    daily_vitals = relationship("DailyVital", back_populates="patient", cascade="all, delete-orphan")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    disease_type = Column(String(64), nullable=False, default="general")
    risk_score = Column(Float, nullable=False)
    result = Column(String(64), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_data = Column(Text, nullable=True)  # store serialized JSON
    created_by = Column(String(64), nullable=True)

    patient = relationship("Patient", back_populates="predictions")


class DailyVital(Base):
    __tablename__ = "daily_vitals"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    taken_at = Column(DateTime, default=datetime.utcnow, index=True)
    systolic_bp = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    heart_rate = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    created_by = Column(String(64), nullable=True)

    patient = relationship("Patient", back_populates="daily_vitals")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="doctor")
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient")
    sessions = relationship("SessionToken", back_populates="user", cascade="all, delete-orphan")


class SessionToken(Base):
    __tablename__ = "session_tokens"

    token = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    receiver_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    read_at = Column(DateTime, nullable=True)


# ---------------- SESSION UTILS ----------------
@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(seed_csv: str | None = None, seed_users_yaml: str | None = "data/users.yaml") -> None:
    """Create tables and optionally seed patients from CSV."""
    Base.metadata.create_all(bind=engine)

    if seed_csv and Path(seed_csv).exists():
        with session_scope() as session:
            patient_exists = session.execute(select(Patient).limit(1)).first()
            if not patient_exists:
                import pandas as pd
                import uuid

                df = pd.read_csv(seed_csv)
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

    if seed_users_yaml:
        seed_users_from_yaml(seed_users_yaml)


def seed_users_from_yaml(yaml_path: str) -> None:
    path = Path(yaml_path)
    if not path.exists():
        return
    try:
        import yaml
        from yaml.loader import SafeLoader
    except ImportError:
        return

    from .auth import hash_password  # local import to avoid circular dependency

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.load(fh, Loader=SafeLoader) or {}

    users = config.get("credentials", {}).get("usernames", {})
    if not users:
        return

    with session_scope() as session:
        for username, info in users.items():
            exists = session.execute(select(User).where(User.username == username)).first()
            if exists:
                continue
            plain = info.get("password_plain")
            password_hash = info.get("password")
            if plain and not password_hash:
                password_hash = hash_password(plain)
            elif password_hash and not password_hash.startswith("$2b$"):
                # Unknown format; re-hash if plain provided
                password_hash = hash_password(password_hash)
            user = User(
                username=username,
                name=info.get("name", username),
                email=info.get("email"),
                password_hash=password_hash or hash_password("password"),
                role=info.get("role", "doctor"),
            )
            session.add(user)
