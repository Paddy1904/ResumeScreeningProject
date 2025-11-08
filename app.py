import os
import json
import traceback
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# CONFIG
# ------------------------------
TFIDF_PATH = "tfidf_vectorizer.pkl"
REGRESSOR_PATH = "resume_regressor.pkl"
KMEANS_PATH = "kmeans.pkl"
DATA_CSV = "resume_automation_screening.csv"

TOP_K = 10

# ------------------------------
# FLASK INIT
# ------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ------------------------------
# LOAD ML ARTIFACTS
# ------------------------------
try:
    vectorizer = joblib.load(TFIDF_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    print("✅ Model & Vectorizer Loaded Successfully")

    kmeans = joblib.load(KMEANS_PATH) if os.path.exists(KMEANS_PATH) else None
except Exception as e:
    print("❌ Model/vectorizer loading error:", e)
    vectorizer = None
    regressor = None
    kmeans = None


# ------------------------------
# UTIL FUNCTIONS
# ------------------------------
def clean(text: str) -> str:
    return str(text).lower().strip()


def normalize(arr):
    if arr.max() - arr.min() < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def rank_resumes_for_jd(jd_text, candidate_texts, skills=None, weight_cos=0.7, weight_skill=0.3):
    jd = clean(jd_text)
    candidates = [clean(t) for t in candidate_texts]

    jd_vec = vectorizer.transform([jd])
    cand_vec = vectorizer.transform(candidates)

    # cosine similarity
    cos_sim = cosine_similarity(jd_vec, cand_vec).flatten()

    # skill matching score
    skill_scores = np.zeros(len(candidate_texts))
    if skills:
        low_skills = [s.lower() for s in skills]
        for i, text in enumerate(candidates):
            skill_scores[i] = sum(1 for s in low_skills if s in text) / len(low_skills)

    cos_norm = normalize(cos_sim)
    skill_norm = normalize(skill_scores)

    final_score = (weight_cos * cos_norm) + (weight_skill * skill_norm)
    final_score = final_score.astype(float)

    # regression model score (optional)
    try:
        predicted_scores = regressor.predict(cand_vec)
    except:
        predicted_scores = [None] * len(candidate_texts)

    results = []
    for i in range(len(candidate_texts)):
        results.append({
            "resume_text": candidate_texts[i],
            "final_score": float(final_score[i]),
            "cos_sim": float(cos_sim[i]),
            "skill_score": float(skill_scores[i]),
            "predicted_suitability": float(predicted_scores[i]) if predicted_scores[i] else None,
            "cluster_label": int(kmeans.predict(cand_vec[i])[0]) if kmeans else None
        })

    results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return results_sorted


# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ✅ Predict resume score from uploaded file
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No resume uploaded"}), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            return jsonify({"error": "Empty file received"}), 400

        resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
        text_transformed = vectorizer.transform([resume_text])

        score = regressor.predict(text_transformed)[0]

        return jsonify({"score": float(score)})
    except Exception as e:
        print("🔥 ERROR DURING PREDICTION:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ✅ Rank resumes based on JD + skills
@app.route("/rank", methods=["POST"])
def rank_endpoint():
    try:
        data = request.get_json()
        jd = data.get("job_description", "")
        candidates = data.get("candidate_texts", [])
        skills = data.get("skills", [])
        top_k = int(data.get("top_k", TOP_K))

        if not jd:
            return jsonify({"error": "job_description required"}), 400
        if not candidates:
            return jsonify({"error": "candidate_texts required"}), 400

        results = rank_resumes_for_jd(jd, candidates, skills)

        return jsonify({"top_k": results[:top_k]})

    except Exception as e:
        print("🔥 ERROR IN /rank:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
