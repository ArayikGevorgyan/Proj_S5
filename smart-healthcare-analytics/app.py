import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional

from src.data_preparation import load_data, clean_data, merge_uploaded_dataset
from src.stats_analysis import descriptive_statistics, correlation_matrix
from src.graph_model import create_feature_graph, visualize_graph
from src.ml_model import (
    load_model, predict_risk, train_models,
    explain_prediction, train_multi_disease_models, load_multi_model,
    encode_categoricals, get_feature_importance
)
from src.patient_database import add_patient, get_all_patients, search_patients, save_prediction_log, get_prediction_history
from src.recommendations import generate_recommendations
from src.report_generator import generate_patient_report
from src.analytics import (
    get_patient_history as analytics_patient_history,
    plot_risk_trend,
    correlation_over_time,
    cluster_patients,
)
from src.chatbot import explain_patient_results, chat_with_ai
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

st.title("🩺 Smart Healthcare Analytics System")
st.write("Predict, manage, and analyze patient health risks using AI.")

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
    "📈 Risk Trend Analysis",
    "🧩 Cluster Analysis",
    "📊 Model Comparison",
    "🔁 Retrain Model",
    "📤 Export Report",
    "🔗 Graph Visualization",
    "💬 AI Chatbot",
]
PATIENT_MENU = [
    "🏠 Home",
    "🧠 AI Prediction",
    "📈 Risk Trend Analysis",
    "📤 Export Report",
    "💬 AI Chatbot",
]

menu_options = DOCTOR_MENU if current_user.role == "doctor" else PATIENT_MENU
menu = st.sidebar.selectbox("Select Section", menu_options)

# ------------------- LOAD DATA -------------------
df = load_data("data/sample_patients.csv")
df_clean = clean_data(df)

# ------------------- HOME -------------------
if menu == "🏠 Home":
    st.subheader("Project Overview")
    st.write("""
    This system assists healthcare professionals in identifying chronic disease risks (e.g., diabetes, heart disease, stroke)
    through AI-driven analysis, patient management, and personalized recommendations.
    """)

    #st.image("docs/plots/feature_graph.png", caption="AI-Powered Health Insights", use_column_width=True)
    st.success("Use the sidebar to navigate through patient data, AI predictions, and reports.")

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
        updated_df = merge_uploaded_dataset(uploaded_file)
        st.success(f"File ingested successfully. Dataset now has {len(updated_df)} records.")
        st.dataframe(updated_df.tail())
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

            # Explainable AI
            shap_path = explain_prediction(model, scaler, input_data)
            if shap_path and os.path.exists(shap_path):
                st.write("### Explainable AI (Feature Importance)")
                st.image(shap_path, caption="SHAP Explanation Plot")

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
                    "input_data": input_data,
                    "shap_path": shap_path if shap_path and os.path.exists(shap_path) else None
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

# ------------------- CLUSTER ANALYSIS -------------------
elif menu == "🧩 Cluster Analysis":
    require_doctor()
    st.subheader("Patient Cluster Analysis")
    if df_clean is None or df_clean.empty:
        st.warning("Dataset unavailable for clustering.")
    else:
        k = st.slider("Number of clusters", min_value=2, max_value=6, value=3)
        cluster_data = cluster_patients(df_clean, n_clusters=k)
        if not cluster_data:
            st.info("Unable to compute clusters with the current dataset.")
        else:
            if cluster_data.get("plot_path"):
                st.image(cluster_data["plot_path"], caption="PCA scatter with cluster labels")
            st.write("### Cluster Distribution")
            st.dataframe(cluster_data["summary"])
            st.write("### Cluster Centroids (feature space)")
            st.dataframe(cluster_data["centroids"])

# ------------------- MODEL COMPARISON -------------------
elif menu == "📊 Model Comparison":
    require_doctor()
    st.subheader("Model Performance Comparison")
    results, _ = train_models(df_clean)
    results_df = pd.DataFrame([
        {"Model": name, "Accuracy": info["accuracy"], "ROC AUC": info["roc_auc"]}
        for name, info in results.items()
    ])
    st.dataframe(results_df)
    st.bar_chart(results_df.set_index("Model")[["Accuracy", "ROC AUC"]])
    st.write("### ROC Curves")
    for name, info in results.items():
        if "roc_curve" in info:
            st.image(info["roc_curve"], caption=f"{name} ROC Curve")

    st.markdown("---")
    st.write("### Feature Importance Insights")
    try:
        model_bundle, scaler = load_model()
    except FileNotFoundError:
        st.info("Train the default model to unlock feature importance charts.")
    else:
        encoded_df = encode_categoricals(df_clean.copy())
        feature_cols = [col for col in encoded_df.columns if col != "Disease"]
        if feature_cols:
            sample_features = encoded_df[feature_cols]
            sample_features = sample_features.head(min(200, len(sample_features)))
            if scaler is not None:
                scaled_values = scaler.transform(sample_features)
                sample_scaled = pd.DataFrame(scaled_values, columns=feature_cols)
            else:
                sample_scaled = sample_features

            importance = get_feature_importance(model_bundle, feature_cols, sample_scaled)

            if importance["model_importance"]:
                imp_df = pd.DataFrame(importance["model_importance"], columns=["Feature", "Importance"]).set_index("Feature")
                st.bar_chart(imp_df, height=300)
            if importance["shap_importance"]:
                shap_df = pd.DataFrame(importance["shap_importance"], columns=["Feature", "SHAP Value"]).set_index("Feature")
                st.bar_chart(shap_df, height=300)
        else:
            st.info("Insufficient numeric features to compute importance.")

# ------------------- RETRAIN MODEL -------------------
elif menu == "🔁 Retrain Model":
    require_doctor()
    st.subheader("Retrain Machine Learning Models")
    if st.button("🚀 Retrain Now"):
        with st.spinner("Training models..."):
            results, _ = train_models(df_clean)
        st.success("✅ Retraining complete.")
        st.json(results)

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
                    "input_data": input_data,
                    "shap_path": None
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

# ------------------- GRAPH VISUALIZATION -------------------
elif menu == "🔗 Graph Visualization":
    require_doctor()
    st.subheader("Health Metric Correlation Network")
    G = create_feature_graph(df_clean)
    visualize_graph(G, output_path="docs/plots/feature_graph.png")
    st.image("docs/plots/feature_graph.png", caption="Feature Correlation Graph")

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
