from flask import Flask, request, jsonify
import os

# ---- Import project modules ----
from src.data_preparation import load_data, clean_data, transform_data, summarize_data
from src.stats_analysis import descriptive_statistics, correlation_matrix, hypothesis_tests
from src.graph_model import create_feature_graph, analyze_graph, visualize_graph
from src.ml_model import (
    train_models, load_model, predict_risk,
    train_multi_disease_models, load_multi_model,
    explain_prediction
)

# ---- Initialize Flask app ----
app = Flask(__name__)

DATA_PATH = "data/sample_patients.csv"


# ---------------- HOME ----------------
@app.route('/')
def home():
    """Welcome message and available routes."""
    return jsonify({
        "message": "Smart Healthcare Analytics System API is running 🚀",
        "routes": {
            "/analyze": "Perform statistical and graph analysis",
            "/train": "Train and evaluate AI models",
            "/train_multi": "Train models for multiple diseases",
            "/predict": "Predict disease risk (POST JSON)",
            "/predict_multi/<disease>": "Predict risk for selected disease model",
            "/compare_models": "Compare model performance (accuracy, AUC)",
            "/retrain": "Retrain models using latest data",
            "/explain": "Generate SHAP explainability plot (POST JSON)"
        }
    })


# ---------------- ANALYZE DATA ----------------
@app.route('/analyze', methods=['GET'])
def analyze():
    """Perform data analysis: descriptive stats, correlations, hypothesis test, graph metrics."""
    try:
        df = load_data(DATA_PATH)
        df_clean = clean_data(df)

        # --- Stats & EDA ---
        desc = descriptive_statistics(df_clean)
        corr = correlation_matrix(df_clean)
        hypo = hypothesis_tests(df_clean)
        summary = summarize_data(df_clean)

        # --- Graph Analysis ---
        G = create_feature_graph(df_clean, corr_threshold=0.5)
        graph_metrics = analyze_graph(G)
        visualize_graph(G)

        response = {
    "status": "success",
    "summary_statistics": desc,
    "correlation_matrix": corr,
    "hypothesis_test": hypo,
    "data_summary": summary,
    "graph_metrics": graph_metrics,
    "graph_image": "docs/plots/feature_graph.png"
}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- TRAIN MODEL ----------------
@app.route('/train', methods=['GET'])
def train_model():
    """Train multiple models and save the best one."""
    try:
        df = load_data(DATA_PATH)
        df_clean = clean_data(df)
        results, best_model = train_models(df_clean, target_column="Disease")

        best_acc = max(v["accuracy"] for v in results.values())
        best_model_name = max(results, key=lambda m: results[m]["accuracy"])

        return jsonify({
            "status": "training completed ✅",
            "models": results,
            "best_model": best_model_name,
            "best_model_accuracy": best_acc,
            "saved_model": "models/model_bundle.pkl"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- MODEL COMPARISON ----------------
@app.route('/compare_models', methods=['GET'])
def compare_models():
    """Train and compare Logistic, Decision Tree, and Random Forest models."""
    try:
        df = load_data(DATA_PATH)
        df_clean = clean_data(df)
        results, _ = train_models(df_clean, target_column="Disease")

        comparison = [
            {"model": name,
             "accuracy": info["accuracy"],
             "roc_auc": info["roc_auc"],
             "roc_curve": info["roc_curve"]}
            for name, info in results.items()
        ]

        return jsonify({
            "status": "comparison completed ✅",
            "comparison_results": comparison
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RETRAIN MODEL ----------------
@app.route('/retrain', methods=['POST', 'GET'])
def retrain_model():
    """Retrain model(s) with latest data."""
    try:
        df = load_data(DATA_PATH)
        df_clean = clean_data(df)
        results, _ = train_models(df_clean, target_column="Disease")

        return jsonify({
            "status": "retraining completed 🔁",
            "models": results,
            "best_model_saved": "models/model_bundle.pkl"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- MULTI-DISEASE TRAINING ----------------
@app.route('/train_multi', methods=['GET'])
def train_multi():
    """Train separate models for multiple diseases (requires separate datasets)."""
    try:
        datasets = {}
        for name in ["diabetes", "heart", "stroke"]:
            path = f"data/{name}_data.csv"
            if os.path.exists(path):
                datasets[name] = clean_data(load_data(path))

        if not datasets:
            return jsonify({"error": "No multi-disease datasets found in /data/ folder."}), 400

        results = train_multi_disease_models(datasets)
        return jsonify({
            "status": "multi-disease training completed ✅",
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease risk using the default trained model."""
    try:
        data = request.get_json()
        model, scaler = load_model()
        prediction = predict_risk(model, scaler, data)
        return jsonify({
            "input": data,
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- MULTI-DISEASE PREDICTION ----------------
@app.route('/predict_multi/<disease>', methods=['POST'])
def predict_multi(disease):
    """Predict using disease-specific model (diabetes, heart, stroke)."""
    try:
        data = request.get_json()
        model, scaler = load_multi_model(disease)
        if model is None:
            return jsonify({"error": f"Model for {disease} not found. Run /train_multi first."}), 400

        prediction = predict_risk(model, scaler, data)
        return jsonify({
            "disease": disease,
            "input": data,
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- EXPLAIN (SHAP) ----------------
@app.route('/explain', methods=['POST'])
def explain():
    """Generate SHAP explanation plot for a given patient input."""
    try:
        data = request.get_json()
        model, scaler = load_model()
        shap_path = explain_prediction(model, scaler, data)
        return jsonify({
            "status": "explanation generated ✅",
            "plot_path": shap_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- DATA SUMMARY ----------------
@app.route('/summary', methods=['GET'])
def data_summary():
    """Functional summary from data_preparation."""
    try:
        df = load_data(DATA_PATH)
        df_clean = clean_data(df)
        df_transformed = transform_data(df_clean)
        stats = summarize_data(df_transformed)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- MAIN ENTRY ----------------
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("docs/plots", exist_ok=True)
    app.run(debug=True, port=5000)
