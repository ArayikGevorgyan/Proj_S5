import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional

from src.data_preparation import load_data, clean_data, merge_uploaded_dataset
from src.stats_analysis import descriptive_statistics, correlation_matrix
# NOTE: graph visualization removed but helpers kept for future re-enable
from src.ml_model import (
    load_model, predict_risk, train_models,
    train_multi_disease_models, load_multi_model,
)
from src.patient_database import (
    add_patient,
    get_all_patients,
    search_patients,
    save_prediction_log,
    get_prediction_history,
    add_daily_vital,
)
from src.recommendations import generate_recommendations, generate_vitals_recommendations
from src.report_generator import generate_patient_report
from src.analytics import (
    get_patient_history as analytics_patient_history,
    plot_risk_trend,
    correlation_over_time,
    get_patient_vitals_history,
    plot_vitals_trend,
)
from src.chatbot import explain_patient_results, chat_with_ai
from src.messaging import send_message, get_conversation
from src.database import init_db
from src.auth import (
    authenticate_user,
    register_doctor_account,
    register_patient_account,
    get_patient_record_for_user,
    create_session_token,
    get_user_by_token,
    invalidate_session_token,
    AuthenticatedUser,
    list_users_by_role,
)


# ------------------- PAGE CONFIG -------------------
init_db(seed_csv="data/patients_db.csv", seed_users_yaml="data/users.yaml")
st.set_page_config(page_title="Smart Healthcare Analytics System", layout="wide")

if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None
if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = None
if "patient_predictions" not in st.session_state:
    st.session_state["patient_predictions"] = {}


def _set_current_user(user: AuthenticatedUser):
    st.session_state["auth_user"] = {
        "id": user.id,
        "username": user.username,
        "name": user.name,
        "role": user.role,
        "email": user.email,
        "patient_db_id": user.patient_db_id,
        "patient_public_id": user.patient_public_id,
    }


def _get_current_user() -> Optional[AuthenticatedUser]:
    data = st.session_state.get("auth_user")
    if data:
        return AuthenticatedUser(**data)

    token = st.session_state.get("auth_token")
    if not token:
        token = st.query_params.get("session")
        if isinstance(token, list):
            token = token[0]

    if token:
        user = get_user_by_token(token)
        if user:
            _set_current_user(user)
            st.session_state["auth_token"] = token
            return user
    return None


def _render_auth_portal():
    st.title("🩺 Smart Healthcare Analytics System")
    st.write("Please log in or register to continue.")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            user = authenticate_user(login_username.strip(), login_password)
            if user:
                _set_current_user(user)
                token = create_session_token(user.id)
                st.session_state["auth_token"] = token
                st.query_params["session"] = token
                st.success(f"Welcome back, {user.name}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        role = st.radio("Register as", ["Doctor", "Patient"], horizontal=True)
        if role == "Doctor":
            with st.form("register_doctor_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Doctor Account")
            if submitted:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    try:
                        register_doctor_account(name=name, email=email, username=username, password=password)
                    except ValueError as err:
                        st.error(str(err))
                    else:
                        st.success("Account created! Please log in.")
        else:
            with st.form("register_patient_form"):
                name = st.text_input("Full Name", key="patient_name")
                email = st.text_input("Email", key="patient_email")
                username = st.text_input("Username", key="patient_username")
                password = st.text_input("Password", type="password", key="patient_password")
                confirm = st.text_input("Confirm Password", type="password", key="patient_confirm")
                age = st.number_input("Age", 1, 100, 30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
                glucose = st.number_input("Glucose", 50, 250, 100)
                bp = st.number_input("Blood Pressure", 50, 180, 80)
                cholesterol = st.number_input("Cholesterol", 100, 350, 200)
                submitted = st.form_submit_button("Create Patient Account")
            if submitted:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    try:
                        register_patient_account(
                            name=name,
                            email=email,
                            username=username,
                            password=password,
                            age=age,
                            gender=gender,
                            bmi=bmi,
                            glucose=glucose,
                            blood_pressure=bp,
                            cholesterol=cholesterol,
                        )
                    except ValueError as err:
                        st.error(str(err))
                    else:
                        st.success("Patient account created! Please log in.")

    st.stop()


current_user = _get_current_user()
if current_user is None:
    _render_auth_portal()

st.sidebar.write(f"👤 Logged in as **{current_user.name}** ({current_user.role.title()})")
if st.sidebar.button("Logout"):
    token = st.session_state.get("auth_token") or st.query_params.get("session")
    if isinstance(token, list):
        token = token[0]
    if token:
        invalidate_session_token(token)
    st.session_state["auth_user"] = None
    st.session_state["auth_token"] = None
    if "session" in st.query_params:
        del st.query_params["session"]
    st.rerun()

user_prediction_cache = st.session_state["patient_predictions"].setdefault(current_user.username, {})


def require_doctor():
    if current_user.role != "doctor":
        st.error("This section is available to doctors only.")
        st.stop()

# ------------------- SIDEBAR MENU -------------------
DOCTOR_MENU = [
    "🏠 Home",
    "📋 Patient Database",
    "📊 Data Analytics",
    "📂 Upload Data",
    "🧠 AI Prediction",
    "❤️ Daily Vitals",
    "📈 Risk Trend Analysis",
    "📤 Export Report",
    "📨 Messages",
    "💬 AI Chatbot",
]
PATIENT_MENU = [
    "🏠 Home",
    "🧠 AI Prediction",
    "❤️ Daily Vitals",
    "📈 Risk Trend Analysis",
    "📤 Export Report",
    "📨 Messages",
    "💬 AI Chatbot",
]

menu_options = DOCTOR_MENU if current_user.role == "doctor" else PATIENT_MENU
menu = st.sidebar.selectbox("Select Section", menu_options)

# ------------------- LOAD DATA -------------------
df = load_data("data/sample_patients.csv")
df_clean = clean_data(df)

# ------------------- HOME -------------------
if menu == "🏠 Home":
    st.markdown(
        """
        <h1 style="text-align:center; margin-top:0.5em;">
            🩺 Smart Healthcare Analytics System
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <h3 style="text-align:center; font-weight:400; margin-top:0.2em;">
            Predict, manage, and analyze patient health risks using AI.
        </h3>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align:center; margin-top:3em; font-size:1.05rem;">
            This system assists healthcare professionals in identifying chronic disease risks
            (e.g., diabetes, heart disease, stroke) through AI-driven analysis, patient management,
            and personalized recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="position:fixed;
                    bottom:2rem;
                    left:0;
                    right:0;
                    display:flex;
                    justify-content:center;
                    z-index:100;">
            <div style="background:#d4f5d0;
                        border-radius:12px;
                        padding:0.9em 1.8em;
                        font-size:1rem;
                        font-weight:600;
                        color:#1f4d1f;
                        box-shadow:0 2px 6px rgba(0,0,0,0.08);">
                Use the sidebar to navigate through patient data, AI predictions, and reports.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------- PATIENT DATABASE -------------------
elif menu == "📋 Patient Database":
    require_doctor()
    st.subheader("Patient Records Management")

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("🔍 Search by Name or ID", "")
    with col2:
        with st.expander("➕ Add New Patient", expanded=False):
            with st.form("add_patient_form"):
                st.write("### Add New Patient")
                name = st.text_input("Full Name")
                age = st.number_input("Age", 1, 100, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
                glucose = st.number_input("Glucose", 50, 250, 110)
                bp = st.number_input("Blood Pressure", 50, 180, 80)
                cholesterol = st.number_input("Cholesterol", 100, 350, 200)
                submitted = st.form_submit_button("Save Patient")
                if submitted:
                    patient_data = {
                        "Name": name.strip(), "Age": age, "Gender": gender,
                        "BMI": bmi, "Glucose": glucose,
                        "BloodPressure": bp, "Cholesterol": cholesterol
                    }
                    if patient_data["Name"]:
                        add_patient(patient_data)
                        st.success(f"✅ Patient '{name}' added successfully!")
                    else:
                        st.error("Please provide the patient's full name before saving.")

    if query:
        patients = search_patients(query)
    else:
        patients = get_all_patients()

    if patients is not None and not patients.empty:
        st.dataframe(patients)

        st.markdown("### Daily Vitals Overview")
        patient_lookup = {}
        options = []
        for _, row in patients.iterrows():
            label = f"{row['Name']} ({row['PatientID']})"
            patient_lookup[label] = row.to_dict()
            options.append(label)

        selected_label = st.selectbox("Select patient to view vitals", options)
        selected_patient = patient_lookup[selected_label]
        selected_id = selected_patient.get("PatientID")

        vitals_df = get_patient_vitals_history(selected_id)
        if vitals_df is None or vitals_df.empty:
            st.info("No daily vitals recorded yet for this patient.")
        else:
            vitals_df_display = vitals_df.copy()
            vitals_df_display["TakenAt"] = pd.to_datetime(vitals_df_display["TakenAt"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(vitals_df_display.tail(20))
            trend_path = plot_vitals_trend(vitals_df, selected_id, selected_patient.get("Name"))
            if trend_path:
                st.image(trend_path, caption="Daily blood pressure and heart-rate trend")
    else:
        st.warning("No patient records found.")

# ------------------- DATA ANALYTICS -------------------
elif menu == "📊 Data Analytics":
    require_doctor()
    st.subheader("Exploratory Data Analysis")
    stats = descriptive_statistics(df_clean)
    st.write("### Descriptive Statistics")
    st.dataframe(stats)

    st.write("### Correlation Heatmap")
    corr_path = "docs/plots/correlation_heatmap.png"
    correlation_matrix(df_clean, output_path=corr_path)
    st.image(corr_path, caption="Correlation Heatmap")

    st.write("### Rolling Correlation Explorer")
    total_records = len(df_clean)
    if total_records >= 5:
        default_value = min(25, total_records)
        upto = st.slider(
            "Records up to",
            min_value=5,
            max_value=total_records,
            value=default_value,
            help="Adjust to see how feature correlations evolve as more records are considered."
        )
        corr_df, rolling_path = correlation_over_time(df_clean, upto_records=upto)
        if rolling_path:
            st.image(rolling_path, caption=f"Rolling correlation (first {upto} records)")
            st.dataframe(corr_df)
    else:
        st.info("Add more records to enable rolling correlation analysis.")

# ------------------- DATA UPLOAD -------------------
elif menu == "📂 Upload Data":
    require_doctor()
    st.subheader("Upload New Patient Dataset")
    st.write("Upload CSV files to extend the analytics dataset. Files will be stored under `data/uploads/` for auditing.")

    uploaded_file = st.file_uploader("Select CSV file", type=["csv"])
    if uploaded_file is not None:
        updated_df, new_records = merge_uploaded_dataset(uploaded_file)
        st.success(f"File ingested successfully. Dataset now has {len(updated_df)} records.")
        st.dataframe(updated_df.tail())

        # Persist uploaded patients into the primary database for availability elsewhere in the app.
        if not new_records.empty:
            upload_label = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            added_count = 0
            for idx, row in new_records.reset_index(drop=True).iterrows():
                patient_payload = {
                    "Name": str(row.get("Name") or f"Uploaded Patient {upload_label}-{idx + 1}"),
                    "Age": row.get("Age"),
                    "Gender": row.get("Gender", "Unknown"),
                    "BMI": row.get("BMI"),
                    "Glucose": row.get("Glucose"),
                    "BloodPressure": row.get("BloodPressure"),
                    "Cholesterol": row.get("Cholesterol")
                }
                add_patient(patient_payload)
                added_count += 1

            st.info(f"{added_count} patient record(s) added to the Patient Database from the uploaded file.")
    else:
        st.info("Awaiting file upload...")

# ------------------- AI PREDICTION -------------------
elif menu == "🧠 AI Prediction":
    st.subheader("AI-Powered Disease Risk Prediction")

    patient_lookup = {}
    selected_patient = None

    if current_user.role == "doctor":
        patients_df = get_all_patients()
        patient_options = ["Manual Entry"]
        if patients_df is not None and not patients_df.empty:
            for _, row in patients_df.iterrows():
                label = f"{row['Name']} ({row['PatientID']})"
                patient_lookup[label] = row.to_dict()
                patient_options.append(label)

        selected_patient_label = st.selectbox("Select Patient", patient_options)
        selected_patient = patient_lookup.get(selected_patient_label)
    else:
        selected_patient = get_patient_record_for_user(current_user)
        if not selected_patient:
            st.error("No patient profile linked to your account. Please contact your doctor.")
            st.stop()
        st.info("Using your profile details for prediction.")

    disease_type = st.selectbox(
        "Select Disease Type",
        ["Default (General)", "Diabetes", "Heart", "Stroke"]
    )

    def _numeric_value(key: str, fallback: float):
        if not selected_patient:
            return fallback
        val = selected_patient.get(key, fallback)
        try:
            if pd.isna(val):
                return fallback
        except TypeError:
            pass
        try:
            return float(val)
        except (TypeError, ValueError):
            return fallback

    gender_default = (selected_patient or {}).get("Gender", "Male")
    if gender_default not in ["Male", "Female"]:
        gender_default = "Male"

    st.write("Enter patient details (editable):")
    if selected_patient:
        st.caption(f"Loaded details for {selected_patient.get('Name')} (ID: {selected_patient.get('PatientID')}).")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, int(_numeric_value("Age", 40)))
        bmi = st.number_input("BMI", 10.0, 50.0, float(_numeric_value("BMI", 25.0)))
        glucose = st.number_input("Glucose", 50, 250, int(_numeric_value("Glucose", 110)))
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if gender_default == "Male" else 1)
    with col2:
        blood_pressure = st.number_input("Blood Pressure", 50, 180, int(_numeric_value("BloodPressure", 80)))
        cholesterol = st.number_input("Cholesterol", 100, 350, int(_numeric_value("Cholesterol", 200)))
        smoking = st.selectbox("Smoking", ["Yes", "No"], index=1)
        activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"], index=1)

    if st.button("Predict Risk"):
        gender_val = 1 if gender == "Male" else 0
        smoking_val = 1 if smoking == "Yes" else 0
        activity_val = {"Low": 0, "Medium": 1, "High": 2}[activity]

        input_data = {
            "Age": age, "BMI": bmi, "Glucose": glucose,
            "BloodPressure": blood_pressure, "Cholesterol": cholesterol,
            "Gender": gender_val, "Smoking": smoking_val,
            "PhysicalActivity": activity_val
        }

        if disease_type == "Default (General)":
            model, scaler = load_model()
        else:
            model, scaler = load_multi_model(disease_type.lower())

        if model is None:
            st.error(f"Model for {disease_type} not found.")
        else:
            result = predict_risk(model, scaler, input_data)
            result["input_data"] = input_data
            st.success(f"**Prediction:** {result['result']} ({result['risk_score']*100:.1f}% risk)")
            st.progress(result["risk_score"])

            # Recommendations
            st.write("### 💬 Health Recommendations")
            recs = generate_recommendations(input_data, result)
            for r in recs:
                st.info(f"• {r}")

            st.write("### 🤖 AI Medical Assistant")
            if st.button("Explain This Result"):
                with st.spinner("Generating explanation..."):
                    try:
                        explanation = explain_patient_results(
                            input_data=input_data,
                            prediction=result["result"],
                            risk_score=result.get("risk_score"),
                        )
                    except Exception as exc:
                        st.error(f"Failed to generate explanation: {exc}")
                    else:
                        chat_html = f"""
                        <div style="
                        background-color: #eef6ff;
                        padding: 20px;
                        border-radius: 15px;
                        border-left: 6px solid #4a8cf7;
                        font-size: 16px;
                        line-height: 1.6;">
                        <b>🤖 Medical AI Assistant says:</b><br><br>
                        {explanation}
                        </div>
                        """
                        st.markdown(chat_html, unsafe_allow_html=True)

            if current_user.role == "patient":
                patient_id = current_user.patient_public_id
            else:
                patient_id = selected_patient.get("PatientID") if selected_patient else None

            if patient_id:
                user_prediction_cache[patient_id] = {
                    "result": result["result"],
                    "risk_score": result["risk_score"],
                    "disease_type": disease_type,
                    "timestamp": result.get("timestamp") or datetime.utcnow().isoformat(timespec="seconds"),
                    "input_data": input_data
                }
                save_prediction_log(
                    patient_id,
                    disease_type.lower(),
                    result,
                    input_data,
                    created_by=current_user.username,
                )
                if selected_patient:
                    st.info(f"Prediction saved for {selected_patient.get('Name')} and will be included in exported reports.")
                else:
                    st.info("Prediction saved to your patient history.")
            else:
                st.info("Prediction completed for manual entry. Select a saved patient to attach results to their reports.")

# ------------------- DAILY VITALS -------------------
elif menu == "❤️ Daily Vitals":
    st.subheader("Daily Heart Pressure & Vitals")

    if current_user.role == "doctor":
        patients = get_all_patients()
        if patients is None or patients.empty:
            st.warning("No patients found. Add patients in the database section first.")
            st.stop()

        patient_lookup = {}
        options = []
        for _, row in patients.iterrows():
            label = f"{row['Name']} ({row['PatientID']})"
            patient_lookup[label] = row.to_dict()
            options.append(label)

        selected_label = st.selectbox("Select Patient", options)
        patient_row = patient_lookup[selected_label]
        patient_id = patient_row.get("PatientID")
        st.caption(f"Viewing vitals for {patient_row.get('Name')} (ID: {patient_id}).")
    else:
        patient_row = get_patient_record_for_user(current_user)
        if not patient_row:
            st.error("No patient profile available.")
            st.stop()
        patient_id = patient_row.get("PatientID")
        st.info("Recording your own daily vitals.")

    col_form, col_history = st.columns([1, 1.2])

    with col_form:
        if current_user.role == "patient":
            st.markdown("### Add Today's Vitals")
            with st.form("daily_vitals_form"):
                systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=250, value=120)
                diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80)
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=70)
                notes = st.text_area("Notes (optional)", height=80)
                submitted = st.form_submit_button("Save Vitals")

            if submitted:
                add_daily_vital(
                    patient_id=patient_id,
                    systolic_bp=float(systolic),
                    diastolic_bp=float(diastolic),
                    heart_rate=float(heart_rate),
                    notes=notes.strip() or None,
                    created_by=current_user.username,
                )
                latest = {
                    "SystolicBP": systolic,
                    "DiastolicBP": diastolic,
                    "HeartRate": heart_rate,
                }
                st.success("Vitals saved successfully.")
                st.markdown("### 💬 Recommendations for Today")
                for rec in generate_vitals_recommendations(latest):
                    st.info(f"• {rec}")
                st.rerun()
        else:
            st.markdown("### Add Today's Vitals")
            st.info("Doctors can view patient vitals here. Patients record vitals from their own accounts.")

    with col_history:
        st.markdown("### History & Trends")
        vitals_df = get_patient_vitals_history(patient_id)
        if vitals_df is None or vitals_df.empty:
            if current_user.role == "patient":
                st.info("No vitals recorded yet. Add today's measurements to start tracking.")
            else:
                st.info("No vitals recorded yet for this patient.")
        else:
            vitals_df_display = vitals_df.copy()
            vitals_df_display["TakenAt"] = pd.to_datetime(vitals_df_display["TakenAt"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(vitals_df_display.tail(20))
            trend_path = plot_vitals_trend(vitals_df, patient_id, patient_row.get("Name"))
            if trend_path:
                st.image(trend_path, caption="Daily blood pressure and heart-rate trend")

# ------------------- RISK TREND ANALYSIS -------------------
elif menu == "📈 Risk Trend Analysis":
    st.subheader("Patient Risk Trend Analysis")
    if current_user.role == "doctor":
        patients = get_all_patients()
        if patients is None or patients.empty:
            st.warning("No patients found. Add patients in the database section first.")
        else:
            patient_lookup = {}
            options = []
            for _, row in patients.iterrows():
                label = f"{row['Name']} ({row['PatientID']})"
                patient_lookup[label] = row.to_dict()
                options.append(label)

            selected_label = st.selectbox("Select Patient", options)
            patient_row = patient_lookup[selected_label]
            patient_id = patient_row.get("PatientID")
    else:
        patient_row = get_patient_record_for_user(current_user)
        patient_id = patient_row.get("PatientID") if patient_row else None
        if not patient_row:
            st.error("No patient profile available.")
            st.stop()
        st.info("Showing your personal risk trend.")

    if patient_id:
        history = analytics_patient_history(patient_id)
        if history.empty:
            st.info("No prediction history found for this patient yet. Run AI Prediction to log entries.")
        else:
            history = history.copy()
            history["RiskScore"] = pd.to_numeric(history["RiskScore"], errors="coerce")
            st.dataframe(history)
            trend_path = plot_risk_trend(history, patient_id, patient_row.get("Name"))
            if trend_path:
                st.image(trend_path, caption="Risk trend over time")
# ------------------- EXPORT REPORT -------------------
elif menu == "📤 Export Report":
    st.subheader("Generate and Download Patient Report")
    predictions_map = user_prediction_cache

    if current_user.role == "doctor":
        patients = get_all_patients()
        if patients is not None and not patients.empty:
            patient_lookup = {}
            options = []
            for _, row in patients.iterrows():
                label = f"{row['Name']} ({row['PatientID']})"
                patient_lookup[label] = row.to_dict()
                options.append(label)

            selected_label = st.selectbox("Select Patient", options)
            patient_row = patient_lookup[selected_label]
        else:
            st.warning("No patients available for report generation.")
            patient_row = None
    else:
        patient_row = get_patient_record_for_user(current_user)
        if not patient_row:
            st.error("No patient profile available.")
            patient_row = None
        else:
            st.info("Generating report for your own records.")

    if patient_row:
        patient_id = patient_row.get("PatientID")
        latest_prediction = predictions_map.get(patient_id)
        if not latest_prediction:
            history_df = get_prediction_history(patient_id)
            if not history_df.empty:
                last = history_df.iloc[-1]
                try:
                    input_data = json.loads(last.get("InputData", "{}"))
                except json.JSONDecodeError:
                    input_data = {}
                latest_prediction = {
                    "result": last.get("Result"),
                    "risk_score": float(last.get("RiskScore", 0) or 0),
                    "disease_type": last.get("DiseaseType"),
                    "timestamp": last.get("Timestamp"),
                    "input_data": input_data
                }

        if latest_prediction:
            st.success(
                f"Latest {latest_prediction.get('disease_type')} prediction "
                f"({latest_prediction.get('timestamp')} UTC): {latest_prediction.get('result')} "
                f"— {latest_prediction.get('risk_score', 0)*100:.1f}% risk"
            )
        else:
            st.info("No stored prediction for this patient. Run the AI Prediction workflow to add one.")

        if st.button("📄 Generate PDF Report"):
            shap_path = latest_prediction.get("shap_path") if latest_prediction else None
            report_path = generate_patient_report(
                patient_row,
                prediction=latest_prediction,
                shap_path=shap_path
            )
            st.success("✅ Report generated successfully.")
            with open(report_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Report",
                    data=file,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf"
                )

# ------------------- DOCTOR-PATIENT MESSAGES -------------------
elif menu == "📨 Messages":
    st.subheader("Doctor–Patient Messages")

    if current_user.role == "doctor":
        partners = list_users_by_role("patient")
        role_label = "patient"
    else:
        partners = list_users_by_role("doctor")
        role_label = "doctor"

    if not partners:
        st.info(f"No {role_label} accounts available to chat with yet.")
    else:
        partner_lookup = {f"{u.name} (@{u.username})": u for u in partners}
        labels = sorted(partner_lookup.keys())
        default_label = labels[0]
        selected_label = st.selectbox("Select conversation", labels, index=labels.index(default_label))
        chat_partner = partner_lookup[selected_label]

        st.markdown(f"Chat between **{current_user.name}** and **{chat_partner.name}**")

        conv = get_conversation(current_user.id, chat_partner.id)
        chat_area = st.container()
        with chat_area:
            if conv is None or conv.empty:
                st.info("No messages yet. Start the conversation below.")
            else:
                for _, row in conv.iterrows():
                    is_me = row["sender_id"] == current_user.id
                    align = "right" if is_me else "left"
                    bubble_color = "#0d6efd" if is_me else "#f1f3f5"
                    text_color = "#ffffff" if is_me else "#212529"
                    border_color = "#084298" if is_me else "#ced4da"
                    label = "You" if is_me else chat_partner.name
                    timestamp = row["timestamp"]
                    ts_str = timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(timestamp, "strftime") else str(timestamp)
                    st.markdown(
                        f"""
                        <div style="
                            max-width:70%;
                            margin:8px 0;
                            padding:10px 14px;
                            border-radius:16px;
                            background-color:{bubble_color};
                            color:{text_color};
                            text-align:left;
                            margin-{ 'left' if is_me else 'right' }:auto;
                            font-size:1rem;
                            border:1px solid {border_color};
                            box-shadow:0 1px 3px rgba(0,0,0,0.12);
                        ">
                            <div style="font-weight:600; margin-bottom:4px;">{label}</div>
                            <div style="white-space:pre-wrap; line-height:1.4;">{row["content"]}</div>
                            <div style="font-size:0.75rem; opacity:0.8; margin-top:6px; text-align:right;">{ts_str}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.divider()
        new_message = st.chat_input("Type a message")
        refresh_clicked = st.button("Refresh")

        if new_message and new_message.strip():
            send_message(current_user.id, chat_partner.id, new_message.strip())
            st.rerun()
        elif refresh_clicked:
            st.rerun()

# ------------------- AI CHATBOT -------------------
elif menu == "💬 AI Chatbot":
    st.subheader("AI Health Chatbot")
    st.write("Ask general health questions or get clarification on your reports. Responses are educational, not medical advice.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, msg in st.session_state["chat_history"]:
        bubble_color = "#eef6ff" if role == "assistant" else "#f8f9fb"
        align = "left" if role == "assistant" else "right"
        st.markdown(
            f"""
            <div style="
            background-color:{bubble_color};
            padding:15px;
            border-radius:15px;
            margin:10px 0;
            text-align:{align};
            border-left: 5px solid {'#4a8cf7' if role == 'assistant' else '#ccc'};
            ">
            <b>{'🤖 Assistant' if role == 'assistant' else '🧑 You'}:</b><br>{msg}
            </div>
            """,
            unsafe_allow_html=True,
        )

    user_question = st.text_input("Enter your question")
    if st.button("Send") and user_question.strip():
        st.session_state["chat_history"].append(("user", user_question.strip()))
        with st.spinner("Thinking..."):
            reply = chat_with_ai(st.session_state["chat_history"], user_question.strip())
        st.session_state["chat_history"].append(("assistant", reply))
        st.rerun()
