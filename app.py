# app.py
# Streamlit web app that mirrors the UI flow in your reference video.
# Dependencies: streamlit, pandas, joblib, numpy, matplotlib, seaborn

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# Utility & Load resources
# -----------------------------
DATA_PATH = Path(".") / "cleaned_team_data.csv"
MODEL_LR = Path("models/lr_best.pkl")
MODEL_RF = Path("models/rf_best.pkl")
SCALER = Path("models/scaler.pkl")
PLOTS_DIR = Path("./plots")

@st.cache_resource
def load_resources():
    data = pd.read_csv(DATA_PATH)
    lr = joblib.load(MODEL_LR)
    rf = joblib.load(MODEL_RF)
    scaler = joblib.load(SCALER)
    return data, lr, rf, scaler

data, lr_model, rf_model, scaler = load_resources()

# Basic settings
st.set_page_config(page_title="FIFA 2026 Predictor", layout="wide")
sns.set_style("whitegrid")

# -----------------------------
# Custom CSS for card styling
# -----------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #2b2f77 0%, #6b2fb0 100%);
        color: white;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
        text-align: center;
    }
    .small-muted { color: rgba(255,255,255,0.85); font-size:12px; }
    .big-number { font-size:34px; font-weight:700; margin-top:8px; }
    .confidence-high { color: #006400; font-weight:700; }
    .confidence-low { color: #8b0000; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Top: Title & Metric Cards
# -----------------------------
st.title("🏆 FIFA World Cup 2026 — Finalist & Match Predictor")
st.markdown("Predicting finalists and match outcomes using historical features + ML models.")

col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    st.markdown("<div class='metric-card'><div class='small-muted'>Teams (Loaded)</div>"
                f"<div class='big-number'>{len(data['team'].unique())}</div></div>", unsafe_allow_html=True)
with col2:
    # matches predicted — we show an example derived number (sum of rows where we have history)
    matches_predicted = data.shape[0]
    st.markdown("<div class='metric-card'><div class='small-muted'>Matches (Historic Rows)</div>"
                f"<div class='big-number'>{matches_predicted}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'><div class='small-muted'>Model</div>"
                "<div class='big-number'>LogReg & RF</div></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Section: Qualified Teams (cards grid)
# -----------------------------
st.header("Qualified Teams — Click a team for details")
teams = sorted(data['team'].unique())

# Display teams as small cards (3 columns)
cols = st.columns(6)
col_idx = 0
selected_team = None

for team in teams:
    with cols[col_idx]:
        if st.button(team, key=f"btn_{team}"):
            st.session_state["selected_team"] = team
    col_idx = (col_idx + 1) % 6

# If a team has been selected, show a detailed panel (modal-like)
if "selected_team" in st.session_state:
    t = st.session_state["selected_team"]
    st.sidebar.header(f"Team Details — {t}")
    hist = data[data['team'] == t]
    if hist.empty:
        st.sidebar.markdown("**No historical team rows available.**")
        st.sidebar.write("Default / placeholder stats will be used in predictions.")
    else:
        # Aggregate useful team stats
        rank = hist['rank'].mean()
        points_est = None  # points not in your data; leave as N/A
        win_rate = hist['win_rate'].mean()
        avg_gd = hist['goal_diff_avg'].mean()
        participations = hist['participations'].max()
        st.sidebar.markdown(f"**FIFA Rank (avg):** {rank:.1f}")
        st.sidebar.markdown(f"**Win %:** {win_rate*100:.1f}%")
        st.sidebar.markdown(f"**Goal Difference (avg):** {avg_gd:.2f}")
        st.sidebar.markdown(f"**Participations:** {int(participations)}")
        # Squad info placeholders (video shows avg rating & age)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Squad snapshot**")
        st.sidebar.markdown(f"- Avg. Rating: {65 + (win_rate*30):.1f} (est.)")
        st.sidebar.markdown(f"- Avg. Age: {26 + (1 - win_rate)*6:.1f} yrs (est.)")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Prediction Insights:**")
        st.sidebar.markdown(f"- Rank and goal-difference are key predictors for {t}.")
        if rank < 30:
            st.sidebar.success("Likely strong team based on historical rank.")
        else:
            st.sidebar.info("Mid/low-ranked team — prediction uncertainty higher.")

    if st.sidebar.button("Close team panel"):
        del st.session_state["selected_team"]

st.markdown("---")

# -----------------------------
# Match Predictor Section
# -----------------------------
st.header("Match Predictor")
st.markdown("Select Home & Away teams to predict the match outcome probabilities and scoreline.")

left_col, right_col = st.columns([1, 1])

with left_col:
    home = st.selectbox("Home Team", options=teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
with right_col:
    away = st.selectbox("Away Team", options=teams, index=teams.index("Germany") if "Germany" in teams else 1)

model_choice = st.radio("Choose model used for match probability (probabilities come from classifier):",
                        ("Logistic Regression", "Random Forest"), index=0)

model = lr_model if model_choice == "Logistic Regression" else rf_model

# Helper: build feature vector for a team pair (simple approach using your features)
def team_features_for_match(team_name):
    # If historic data exists, use means; otherwise default values
    hist = data[data['team'] == team_name]
    if hist.empty:
        return {
            "win_rate": 0.30,
            "goal_diff_avg": 0.0,
            "participations": 1,
            "rank": 55
        }
    return {
        "win_rate": float(hist['win_rate'].mean()),
        "goal_diff_avg": float(hist['goal_diff_avg'].mean()),
        "participations": int(hist['participations'].max()),
        "rank": float(hist['rank'].mean())
    }

def match_prediction(home, away, model, scaler, n_score_sim=1000):
    # Convert team stats to the same feature vector you trained on.
    fh = team_features_for_match(home)
    fa = team_features_for_match(away)
    # For match-level probability, build a pairwise feature vector:
    # Here we'll create simple features: difference of metrics (home - away)
    feat = [
        fh['goal_diff_avg'] - fa['goal_diff_avg'],  # goal_diff_avg_diff
        fh['win_rate'] - fa['win_rate'],            # win_rate_diff
        fh['rank'] - fa['rank'],                    # rank_diff
        fh['participations'] - fa['participations'] # participations_diff
    ]
    # Scale using your saved scaler (expecting same order as training)
    X = np.array(feat).reshape(1, -1)
    # NOTE: scaler expects features in original training order; we used goal_diff_avg, win_rate, rank, participations
    # Using the difference vector is a pragmatic choice to estimate likelihood of winning.
    Xs = scaler.transform(np.array([[feat[0], feat[1], (fh['rank']+fa['rank'])/2, (fh['participations']+fa['participations'])/2]]))
    # Our models were trained to predict finalists per team (not match-level). We'll interpret predicted probability
    # as the probability of "team being strong" and combine them into a match probability.
    p_home_strength = model.predict_proba(scaler.transform(pd.DataFrame([[fh['goal_diff_avg'], fh['win_rate'], fh['rank'], fh['participations']]], 
                                                                         columns=['goal_diff_avg','win_rate','rank','participations'])))[:,1][0]
    p_away_strength = model.predict_proba(scaler.transform(pd.DataFrame([[fa['goal_diff_avg'], fa['win_rate'], fa['rank'], fa['participations']]], 
                                                                         columns=['goal_diff_avg','win_rate','rank','participations'])))[:,1][0]
    # Convert strengths to match probabilities via softmax-like scaling
    exp_h = np.exp(p_home_strength)
    exp_a = np.exp(p_away_strength)
    prob_home = exp_h / (exp_h + exp_a)
    prob_away = exp_a / (exp_h + exp_a)
    draw_prob = 0.05  # small fixed draw probability for demonstration; could be tuned

    # Scoreline simulation using Poisson: expected goals derived from strengths & win_rate & goal_diff
    # These lambdas are heuristic and for demo only (not trained).
    # Normalize factors to produce reasonable lambdas
    def lambda_for(team):
        base = 1.0
        lam = base + team['win_rate'] * 1.2 + max(team['goal_diff_avg'], 0) * 0.08 + (60 - team['rank']) * 0.01
        return max(0.2, lam)
    lam_h = lambda_for(fh)
    lam_a = lambda_for(fa)

    # simulate scorelines
    rng = np.random.default_rng(42)
    possible_scores = {}
    for _ in range(n_score_sim):
        gh = rng.poisson(lam_h)
        ga = rng.poisson(lam_a)
        sc = f"{gh}-{ga}"
        possible_scores[sc] = possible_scores.get(sc, 0) + 1
    # convert to percentages and top scorelines
    total_sim = sum(possible_scores.values())
    top_scorelines = sorted([(s, c/total_sim*100) for s, c in possible_scores.items()], key=lambda x: -x[1])[:8]

    confidence_tag = "HIGH CONFIDENCE" if max(prob_home, prob_away) > 0.7 else ("MEDIUM CONFIDENCE" if max(prob_home, prob_away) > 0.55 else "LOW CONFIDENCE")
    return {
        "prob_home": prob_home,
        "prob_away": prob_away,
        "draw_prob": draw_prob,
        "lam_home": lam_h,
        "lam_away": lam_a,
        "top_scorelines": top_scorelines,
        "confidence_tag": confidence_tag
    }

if st.button("Predict Match Outcome"):
    with st.spinner("Running match simulations..."):
        res = match_prediction(home, away, model, scaler, n_score_sim=2000)

    # show results (styled similarly to video)
    left, mid, right = st.columns([1, 1, 1])
    with left:
        st.markdown(f"### {home}")
        st.metric(label="Win Probability", value=f"{res['prob_home']*100:.1f}%")
    with mid:
        st.markdown("### Result")
        winner = home if res['prob_home'] > res['prob_away'] else away
        st.markdown(f"**Predictor Says:** {winner} wins")
        if res['confidence_tag'] == "HIGH CONFIDENCE":
            st.markdown(f"<span class='confidence-high'>{res['confidence_tag']}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='confidence-low'>{res['confidence_tag']}</span>", unsafe_allow_html=True)
    with right:
        st.markdown(f"### {away}")
        st.metric(label="Win Probability", value=f"{res['prob_away']*100:.1f}%")

    st.markdown("**Predicted Score (most-likely):**")
    # pick top one
    top1 = res['top_scorelines'][0]
    st.markdown(f"**{top1[0]}**  — {top1[1]:.1f}% chance (Top scoreline)")

    st.markdown("**Top Scorelines**")
    for sc, pct in res['top_scorelines'][:6]:
        st.write(f"- {sc} : {pct:.1f}%")

    # Save match prediction to CSV (append)
    rec = {
        "home": home, "away": away,
        "prob_home": res['prob_home'], "prob_away": res['prob_away'],
        "top_score": top1[0], "confidence": res['confidence_tag']
    }
    out_file = Path("match_predictions.csv")
    df_rec = pd.DataFrame([rec])
    if out_file.exists():
        df_rec.to_csv(out_file, mode='a', header=False, index=False)
    else:
        df_rec.to_csv(out_file, index=False)
    st.success("Match prediction saved to match_predictions.csv")

st.markdown("---")

# -----------------------------
# Model Evaluation & Methodology
# -----------------------------
st.header("Model Evaluation & Methodology")
st.markdown("""
**Model Type (used in this app):** Logistic Regression (classification of strong teams)
- Random Forest is also available for alternate predictions.

**Training Data:** Historical team-level aggregated rows derived from match history (features: goal_diff_avg, win_rate, rank, participations).

**Input Features (examples):**
- Attack/Defense proxies: `goal_diff_avg`
- Win performance: `win_rate`
- Official ranking: `rank`
- Experience measure: `participations`

**Notes:** The classifier was trained to predict whether a team historically reached a final (binary). For match predictions we convert per-team model probabilities into head-to-head probabilities with a softmax-like scaling; score predictions use a Poisson heuristic using win_rate, rank and goal_diff_avg to produce expected goals and plausible scorelines.
""")

# Show plots if present
col_a, col_b = st.columns(2)
with col_a:
    f_conf_lr = PLOTS_DIR / "confusion_logistic_regression.png"
    if f_conf_lr.exists():
        st.image(str(f_conf_lr), caption="Confusion Matrix — Logistic Regression", use_column_width=True)
    f_roc_lr = PLOTS_DIR / "roc_logistic_regression.png"
    if f_roc_lr.exists():
        st.image(str(f_roc_lr), caption="ROC — Logistic Regression", use_column_width=True)
with col_b:
    f_conf_rf = PLOTS_DIR / "confusion_random_forest.png"
    if f_conf_rf.exists():
        st.image(str(f_conf_rf), caption="Confusion Matrix — Random Forest", use_column_width=True)
    f_roc_rf = PLOTS_DIR / "roc_random_forest.png"
    if f_roc_rf.exists():
        st.image(str(f_roc_rf), caption="ROC — Random Forest", use_column_width=True)

# Feature importance plot
f_feat = PLOTS_DIR / "feature_importance.png"
if f_feat.exists():
    st.image(str(f_feat), caption="Feature Importance (LR vs RF)", use_column_width=True)

st.markdown("**Performance summary (example):** Logistic Regression AUC ~0.78 (good separation on probabilities), Random Forest AUC ~0.70. Models tend to predict non-finalist for the majority class because finalists are rare (class imbalance).")

st.markdown("---")
st.caption("App built for the FIFA 2026 Finalist Prediction assignment — uses your saved models and cleaned dataset.")
