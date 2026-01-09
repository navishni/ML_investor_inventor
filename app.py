import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import plotly.express as px
import math
import joblib
import xgboost as xgb  # ✅ XGBoost instead of CatBoost

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Investor–Idea Recommendation System", layout="wide")
st.title("💡 Investor–Idea Recommendation System")

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect("matching.db")
    investors = pd.read_sql("SELECT * FROM investors", conn)
    ideas = pd.read_sql("SELECT * FROM ideas", conn)
    history = pd.read_sql("SELECT * FROM history", conn)
    conn.close()
    return investors, ideas, history

investors, ideas, history = load_data()
ideas["domain"] = ideas["domain"].str.strip().str.lower()

# -----------------------
# HANDLE FEEDBACK
# -----------------------
if "rating" not in history.columns or history["rating"].sum() == 0:
    history["rating"] = np.random.randint(1, 6, size=len(history))

# -----------------------
# INTERACTION MATRIX
# -----------------------
interaction = history.pivot_table(
    index="investor_id",
    columns="idea_id",
    values="rating",
    aggfunc="sum",
    fill_value=0
)

# -----------------------
# COSINE SIMILARITY
# -----------------------
similarity = cosine_similarity(interaction)
similarity = pd.DataFrame(similarity, index=interaction.index, columns=interaction.index)

# -----------------------
# ✅ NEW: LOAD XGBoost MODEL + ENCODER
# -----------------------
@st.cache_data
def load_model():
    """Load XGBoost model and domain encoder"""
    try:
        model = joblib.load('xgboost_ranker.pkl')
        encoder = joblib.load('domain_encoder.pkl')
        st.success("✅ XGBoost Ranker + Encoder loaded!")
        return {'model': model, 'encoder': encoder}
    except FileNotFoundError:
        st.warning("⚠️ Model files not found. Using rule-based fallback.")
        return None

model_data = load_model()  # ✅ NOW DEFINED!

# -----------------------
# TREND SCORE (API + FALLBACK)
# -----------------------
NEWS_API_KEY = "pub_95c73588d55148f4886bee2270718f33"

@st.cache_data(show_spinner=False)
def fetch_trend_score(domain):
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={domain}&language=en"
        response = requests.get(url, timeout=5).json()
        count = len(response.get("results", []))
        return math.log1p(count)
    except:
        return np.random.uniform(1, 5)

trend_map = {d: fetch_trend_score(d) for d in ideas["domain"].unique()}

def get_trend_score(domain):
    return trend_map.get(domain, np.random.uniform(1, 5))

# -----------------------
# ✅ FIXED RECOMMENDATION FUNCTION (XGBoost)
# -----------------------
def recommend(investor_id, top_n=5):
    rows = []
    for idea_id in interaction.columns:
        idea_domain = ideas.loc[ideas["idea_id"] == idea_id, "domain"].values[0]
        
        # Compute features matching training data
        cf_score = np.dot(similarity.loc[investor_id], interaction[idea_id])
        cf_score = cf_score / (np.linalg.norm(similarity.loc[investor_id]) + 1e-8)
        trend_score = get_trend_score(idea_domain)
        investor_activity = history[history['investor_id'] == investor_id].shape[0]
        idea_popularity = history[history['idea_id'] == idea_id].shape[0]
        
        # Prepare feature vector
        features = pd.DataFrame({
            'cf_score': [cf_score],
            'trend_score': [trend_score],
            'investor_activity': [investor_activity],
            'idea_popularity': [idea_popularity],
            'domain': [idea_domain]
        })
        
        if model_data is not None:  # ✅ FIXED: Use model_data
            # ✅ XGBoost prediction with domain encoding
            features['domain_encoded'] = model_data['encoder'].transform(features['domain'])
            X_pred = features[['cf_score', 'trend_score', 'investor_activity', 'idea_popularity', 'domain_encoded']]
            pred_score = model_data['model'].predict(X_pred)[0]
            final_score = pred_score
            model_used = "XGBoost Ranker"
        else:
            # Fallback to original rule-based
            final_score = 0.7 * cf_score + 0.3 * trend_score
            model_used = "Rule-based"
        
        rows.append([idea_id, idea_domain, final_score, cf_score, trend_score, model_used])
    
    df = pd.DataFrame(
        rows,
        columns=["Idea ID", "Domain", "Final Score", "CF Score", "Trend Score", "Model Used"]
    )
    return df.sort_values("Final Score", ascending=False).head(top_n)

# -----------------------
# PRECISION / RECALL / NDCG @5
# -----------------------
def precision_recall_ndcg_at_k(investor_id, k=5):
    relevant = set(history[history["investor_id"] == investor_id]["idea_id"])
    if len(relevant) == 0:
        return None, None, None
    
    recommended = recommend(investor_id, top_n=k)["Idea ID"].tolist()
    hits = len(set(recommended) & relevant)
    precision = hits / k
    recall = hits / len(relevant)
    
    dcg = sum((1 / np.log2(i + 2) if item in relevant else 0) for i, item in enumerate(recommended))
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    return precision, recall, ndcg

# -----------------------
# STREAMLIT UI (UNCHANGED)
# -----------------------
st.subheader("🎯 Select Investor")
company_names = investors["investor_name"].sort_values().unique()
selected_company = st.selectbox("Select Investor Company", company_names)
available_ids = investors[investors["investor_name"] == selected_company]["investor_id"].values
selected_investor = st.selectbox("Select Investor ID", available_ids)

if selected_investor is not None:
    results = recommend(selected_investor)
    
    st.subheader("📌 Recommended Ideas")
    st.dataframe(results, use_container_width=True)
    
    # -----------------------
# COMPLETE CHARTS SECTION (Replace everything after st.dataframe)
# -----------------------
st.subheader("📌 Recommended Ideas")
st.dataframe(results, use_container_width=True)

# 🥧 FIXED PIE CHART
st.subheader("🥧 Top 5 Ideas Contribution (%)")
results_display = results.copy()
results_display['Final Score'] = np.abs(results_display['Final Score'])
results_display['Final Score'] = results_display['Final Score'] / results_display['Final Score'].sum() * 100

fig_pie = px.pie(
    results_display,
    names="Idea ID",
    values="Final Score",
    title=f"Top 5 Ideas for Investor {selected_investor}",
    hole=0.3,
    color_discrete_sequence=px.colors.sequential.Sunsetdark
)
fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
fig_pie.update_layout(showlegend=True, legend_title="Ideas")
st.plotly_chart(fig_pie, use_container_width=True)

# 📊 ENHANCED BAR CHART
st.subheader("📊 Score Breakdown per Idea")
df_long = results.melt(
    id_vars=["Idea ID", "Domain"],
    value_vars=["CF Score", "Trend Score", "Final Score"],
    var_name="Score Type",
    value_name="Score"
)

fig_bar = px.bar(
    df_long,
    x="Idea ID",
    y="Score",
    color="Score Type",
    barmode="group",
    text="Score",
    color_discrete_map={
        "CF Score": "#1f77b4",      # Blue
        "Trend Score": "#ff7f0e",    # Orange
        "Final Score": "#2ca02c"     # Green
    },
    title="Collaborative Filtering vs Trend vs Final Score"
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(showlegend=True, xaxis_title="Idea ID", yaxis_title="Score")
st.plotly_chart(fig_bar, use_container_width=True)

# 📈 METRICS (unchanged)
precision, recall, ndcg = precision_recall_ndcg_at_k(selected_investor, k=5)
st.subheader("📈 Recommendation Quality")
if precision is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision@5", f"{precision:.3f}")
    col2.metric("Recall@5", f"{recall:.3f}")
    col3.metric("NDCG@5", f"{ndcg:.3f}")
else:
    st.info("⚠️ Add more interaction history for better metrics")

    
    precision, recall, ndcg = precision_recall_ndcg_at_k(selected_investor, k=5)
    st.subheader("📈 Recommendation Quality Metrics")
    if precision is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision@5", f"{precision:.2f}")
        c2.metric("Recall@5", f"{recall:.2f}")
        c3.metric("NDCG@5", f"{ndcg:.2f}")
    else:
        st.info("Not enough data to compute evaluation metrics.")
