"""
World Cup 2026 — Match Predictor
Streamlit app wrapping the Elo + XGBoost prediction engine.
"""

import io
import urllib.request
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="centered",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

  /* Background */
  .stApp { background: #0a0e1a; }

  /* Hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Hero title */
  .hero {
    text-align: center;
    padding: 2.5rem 0 1rem;
  }
  .hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 0.12em;
    color: #ffffff;
    line-height: 1;
    margin: 0;
  }
  .hero .subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    color: #c9a84c;
    text-transform: uppercase;
    margin-top: 0.5rem;
  }

  .created-by {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 400;
    letter-spacing: 0.15em;
    color: #3d4f6e;
    text-transform: uppercase;
    margin-top: 0.4rem;
  }
  .created-by span {
    color: #c9a84c;
    font-weight: 600;
  }

  /* Divider */
  .gold-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, #c9a84c, transparent);
    margin: 1.2rem auto 2rem;
    width: 60%;
  }

  /* Card */
  .card {
    background: #131929;
    border: 1px solid #1e2d47;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
  }

  /* Versus strip */
  .versus-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    margin: 1rem 0 1.4rem;
  }
  .team-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #ffffff;
    letter-spacing: 0.08em;
  }
  .vs-badge {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: #c9a84c;
    background: #1e2d47;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* Elo chips */
  .elo-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 1.2rem;
  }
  .elo-chip {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #8a9bbf;
    text-align: center;
  }
  .elo-chip span {
    display: block;
    font-size: 1.1rem;
    font-weight: 600;
    color: #c9a84c;
  }

  /* Prediction label */
  .pred-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #8a9bbf;
    text-align: center;
    margin-bottom: 0.3rem;
  }
  .pred-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 0.1em;
    color: #ffffff;
    text-align: center;
    margin-bottom: 1.4rem;
  }

  /* Probability strip */
  .prob-strip {
    display: flex;
    justify-content: center;
    gap: 0;
    margin: 1rem 0 1.4rem;
    border: 1px solid #1e2d47;
    border-radius: 10px;
    overflow: hidden;
  }
  .prob-block {
    flex: 1;
    text-align: center;
    padding: 0.9rem 0.5rem;
    background: #0f1627;
    border-right: 1px solid #1e2d47;
    transition: background 0.2s;
  }
  .prob-block:last-child { border-right: none; }
  .draw-block { border-left: 1px solid #1e2d47; }
  .prob-active { background: #1a2540; }
  .prob-team {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8a9bbf;
    margin-bottom: 0.3rem;
  }
  .prob-pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #ffffff;
    letter-spacing: 0.05em;
    line-height: 1;
  }
  .prob-active .prob-pct { color: #c9a84c; }
  .prob-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    color: #3d4f6e;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
  }

  /* Section label */
  .section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 0.8rem;
  }

  /* Disclaimer */
  .disclaimer {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    color: #3d4f6e;
    text-align: center;
    margin-top: 2rem;
    padding-bottom: 2rem;
  }

  /* Selectbox labels */
  label { color: #8a9bbf !important; font-family: 'Inter', sans-serif !important;
          font-size: 0.72rem !important; letter-spacing: 0.12em !important; }

  /* Button */
  .stButton > button {
    background: linear-gradient(135deg, #c9a84c, #a07830) !important;
    color: #0a0e1a !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    margin-top: 0.4rem;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #e0bb60, #c9a84c) !important;
    transform: translateY(-1px);
  }

  /* Metric override */
  [data-testid="metric-container"] { background: #0f1627; border-radius: 8px; padding: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ──────────────────────────────────────────────────────────────────────
ELO_K_BASE = 20
TOURNAMENT_K_MULTIPLIER = {
    "FIFA World Cup": 2.0, "Copa América": 1.5, "UEFA Euro": 1.5,
    "African Cup of Nations": 1.4, "AFC Asian Cup": 1.4,
    "CONCACAF Gold Cup": 1.3, "UEFA Nations League": 1.2,
    "FIFA World Cup qualification": 1.1, "Friendly": 0.8,
}
DEFAULT_K_MULT = 1.0
HOME_ADVANTAGE_ELO = 100
ELO_INIT = 1500
ELO_START_DATE = "1960-01-01"
FORM_WINDOW = 10
RECENCY_HALFLIFE = 180
LABEL_MAP = {0: "Home Win", 1: "Draw", 2: "Away Win"}
FEATURES = [
    "elo_diff_pre", "elo_home_pre", "elo_away_pre",
    "home_form_w_rate", "home_form_avg_gd", "home_form_avg_gf", "home_form_avg_ga", "home_form_n",
    "away_form_w_rate", "away_form_avg_gd", "away_form_avg_gf", "away_form_avg_ga", "away_form_n",
    "h2h_home_wr", "h2h_n", "is_neutral", "is_wc",
]

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def get_k(tournament):
    return ELO_K_BASE * TOURNAMENT_K_MULTIPLIER.get(tournament, DEFAULT_K_MULT)

def margin_multiplier(goal_diff):
    return np.log1p(abs(goal_diff)) + 1

def recency_weight(days_ago, halflife=RECENCY_HALFLIFE):
    return np.exp(-np.log(2) * days_ago / halflife)

def compute_elo_ratings(df_sorted):
    ratings = {}
    records = []
    for _, row in df_sorted.iterrows():
        h, a = row["home_team"], row["away_team"]
        r_h = ratings.get(h, ELO_INIT)
        r_a = ratings.get(a, ELO_INIT)
        neutral = bool(row["neutral"])
        r_h_adj = r_h + (0 if neutral else HOME_ADVANTAGE_ELO)
        E_h = expected_score(r_h_adj, r_a)
        g_h, g_a = int(row["home_score"]), int(row["away_score"])
        S_h = 1.0 if g_h > g_a else (0.0 if g_h < g_a else 0.5)
        S_a = 1 - S_h
        K_adj = get_k(row["tournament"]) * margin_multiplier(g_h - g_a)
        records.append({"idx": row.name, "elo_home_pre": r_h, "elo_away_pre": r_a,
                         "elo_home_adj_pre": r_h_adj, "elo_diff_pre": r_h_adj - r_a})
        ratings[h] = r_h + K_adj * (S_h - E_h)
        ratings[a] = r_a + K_adj * (S_a - (1 - E_h))
    return ratings, pd.DataFrame(records).set_index("idx")

def build_form_features(df_sorted):
    team_history = {}
    records = []
    for _, row in df_sorted.iterrows():
        h, a = row["home_team"], row["away_team"]
        match_date = row["date"]
        def get_form(team):
            hist = team_history.get(team, [])
            if not hist:
                return {"form_w_rate": 0.5, "form_d_rate": 0.2, "form_l_rate": 0.3,
                        "form_avg_gd": 0.0, "form_avg_gf": 1.2, "form_avg_ga": 1.2, "form_n": 0}
            recent = hist[-FORM_WINDOW:]
            weights = [recency_weight((match_date - r["date"]).days) for r in recent]
            tw = sum(weights)
            return {
                "form_w_rate":  sum(w for w, r in zip(weights, recent) if r["result"] == "W") / tw,
                "form_d_rate":  sum(w for w, r in zip(weights, recent) if r["result"] == "D") / tw,
                "form_l_rate":  sum(w for w, r in zip(weights, recent) if r["result"] == "L") / tw,
                "form_avg_gd":  sum(w * r["goal_diff"] for w, r in zip(weights, recent)) / tw,
                "form_avg_gf":  sum(w * r["gf"] for w, r in zip(weights, recent)) / tw,
                "form_avg_ga":  sum(w * r["ga"] for w, r in zip(weights, recent)) / tw,
                "form_n": len(recent),
            }
        h_form, a_form = get_form(h), get_form(a)
        records.append({"idx": row.name,
                         **{f"home_{k}": v for k, v in h_form.items()},
                         **{f"away_{k}": v for k, v in a_form.items()}})
        g_h, g_a = int(row["home_score"]), int(row["away_score"])
        h_res = "W" if g_h > g_a else ("L" if g_h < g_a else "D")
        a_res = "L" if h_res == "W" else ("W" if h_res == "L" else "D")
        for team, res, gf, ga in [(h, h_res, g_h, g_a), (a, a_res, g_a, g_h)]:
            team_history.setdefault(team, []).append(
                {"date": match_date, "result": res, "goal_diff": gf - ga, "gf": gf, "ga": ga})
    return pd.DataFrame(records).set_index("idx")

def build_h2h_features(df_sorted):
    h2h_history = {}
    records = []
    for _, row in df_sorted.iterrows():
        h, a = row["home_team"], row["away_team"]
        key = tuple(sorted([h, a]))
        recent = h2h_history.get(key, [])[-10:]
        if not recent:
            h2h_wr, h2h_n = 0.33, 0
        else:
            wins = sum(1 for r in recent if r["winner"] == h)
            draws = sum(1 for r in recent if r["winner"] == "draw")
            h2h_wr = (wins + 0.5 * draws) / len(recent)
            h2h_n = len(recent)
        records.append({"idx": row.name, "h2h_home_wr": h2h_wr, "h2h_n": h2h_n})
        g_h, g_a = int(row["home_score"]), int(row["away_score"])
        winner = h if g_h > g_a else (a if g_a > g_h else "draw")
        h2h_history.setdefault(key, []).append({"winner": winner})
    return pd.DataFrame(records).set_index("idx")

@st.cache_resource(show_spinner=False)
def load_and_train():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    with urllib.request.urlopen(url) as r:
        df = pd.read_csv(io.StringIO(r.read().decode("utf-8")), parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    df_m = df[df["date"] >= ELO_START_DATE].copy()

    _, elo_f = compute_elo_ratings(df_m)
    df_m = df_m.join(elo_f)
    df_m = df_m.join(build_form_features(df_m))
    df_m = df_m.join(build_h2h_features(df_m))

    def outcome(row):
        return 0 if row["home_score"] > row["away_score"] else (2 if row["home_score"] < row["away_score"] else 1)
    df_m["outcome"] = df_m.apply(outcome, axis=1)
    df_m["is_wc"] = (df_m["tournament"] == "FIFA World Cup").astype(int)
    df_m["is_neutral"] = df_m["neutral"].astype(int)

    df_feat = df_m[FEATURES + ["outcome", "date", "tournament"]].dropna(subset=FEATURES)
    mask = df_feat["date"] < pd.Timestamp("2026-06-01")
    model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="mlogloss", random_state=42, n_jobs=-1)
    model.fit(df_feat[mask][FEATURES].values, df_feat[mask]["outcome"].values, verbose=False)

    final_ratings, _ = compute_elo_ratings(df_m)
    teams = sorted(set(df_m["home_team"].tolist() + df_m["away_team"].tolist()))
    return model, final_ratings, df_m, teams

def get_current_form(team, df_m):
    mask = (df_m["home_team"] == team) | (df_m["away_team"] == team)
    matches = df_m[mask].tail(FORM_WINDOW)
    if len(matches) == 0:
        return {"form_w_rate": 0.5, "form_avg_gd": 0, "form_avg_gf": 1.2, "form_avg_ga": 1.2, "form_n": 0}
    ref = matches["date"].max()
    results = []
    for _, m in matches.iterrows():
        is_home = m["home_team"] == team
        gf = m["home_score"] if is_home else m["away_score"]
        ga = m["away_score"] if is_home else m["home_score"]
        gd = gf - ga
        results.append({"date": m["date"], "result": "W" if gd > 0 else ("L" if gd < 0 else "D"),
                         "gf": gf, "ga": ga, "gd": gd})
    weights = [recency_weight((ref - r["date"]).days) for r in results]
    tw = sum(weights)
    return {
        "form_w_rate":  sum(w for w, r in zip(weights, results) if r["result"] == "W") / tw,
        "form_avg_gd":  sum(w * r["gd"] for w, r in zip(weights, results)) / tw,
        "form_avg_gf":  sum(w * r["gf"] for w, r in zip(weights, results)) / tw,
        "form_avg_ga":  sum(w * r["ga"] for w, r in zip(weights, results)) / tw,
        "form_n": len(results),
    }

def predict_match(home_team, away_team, model, final_ratings, df_m, neutral=True):
    r_h = final_ratings.get(home_team, ELO_INIT)
    r_a = final_ratings.get(away_team, ELO_INIT)
    r_h_adj = r_h + (0 if neutral else HOME_ADVANTAGE_ELO)
    h_form = get_current_form(home_team, df_m)
    a_form = get_current_form(away_team, df_m)
    all_h2h = df_m[
        ((df_m["home_team"] == home_team) & (df_m["away_team"] == away_team)) |
        ((df_m["home_team"] == away_team) & (df_m["away_team"] == home_team))
    ].tail(10)
    if len(all_h2h) == 0:
        h2h_wr, h2h_n = 0.33, 0
    else:
        wins = sum(1 for _, m in all_h2h.iterrows()
                   if (m["home_team"] == home_team and m["home_score"] > m["away_score"]) or
                      (m["away_team"] == home_team and m["away_score"] > m["home_score"]))
        draws = sum(1 for _, m in all_h2h.iterrows() if m["home_score"] == m["away_score"])
        h2h_wr = (wins + 0.5 * draws) / len(all_h2h)
        h2h_n = len(all_h2h)
    X = np.array([[r_h_adj - r_a, r_h, r_a,
                   h_form["form_w_rate"], h_form["form_avg_gd"], h_form["form_avg_gf"], h_form["form_avg_ga"], h_form["form_n"],
                   a_form["form_w_rate"], a_form["form_avg_gd"], a_form["form_avg_gf"], a_form["form_avg_ga"], a_form["form_n"],
                   h2h_wr, h2h_n, int(neutral), 1]])
    probs = model.predict_proba(X)[0]
    return probs, r_h, r_a, h_form, a_form, h2h_n

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>World Cup 2026</h1>
  <div class="subtitle">Match Outcome Predictor · Elo + XGBoost</div>
  <div class="created-by">Created by <span>SSA</span></div>
</div>
<div class="gold-line"></div>
""", unsafe_allow_html=True)

with st.spinner("Loading model & data..."):
    model, final_ratings, df_m, teams = load_and_train()

# ── TEAM SELECTION ─────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Select Teams</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("TEAM 1", teams, index=teams.index("France") if "France" in teams else 0)
with col2:
    away = st.selectbox("TEAM 2", teams, index=teams.index("Senegal") if "Senegal" in teams else 1)

neutral = st.checkbox("Neutral venue (World Cup)", value=True)
predict_btn = st.button("⚽  Predict Match")
st.markdown('</div>', unsafe_allow_html=True)

# ── RESULTS ────────────────────────────────────────────────────────────────────
if predict_btn:
    if home == away:
        st.warning("Please select two different teams.")
    else:
        probs, r_h, r_a, h_form, a_form, h2h_n = predict_match(
            home, away, model, final_ratings, df_m, neutral)

        pred_idx = int(np.argmax(probs))
        pred_label = LABEL_MAP[pred_idx]
        if pred_idx == 0:
            pred_display = f"{home} Win"
        elif pred_idx == 2:
            pred_display = f"{away} Win"
        else:
            pred_display = "Draw"

        # ── VERSUS STRIP
        st.markdown(f"""
        <div class="card">
          <div class="versus-row">
            <div class="team-name">{home}</div>
            <div class="vs-badge">VS</div>
            <div class="team-name">{away}</div>
          </div>
          <div class="elo-row">
            <div class="elo-chip">ELO<span>{round(r_h)}</span></div>
            <div class="elo-chip">DIFFERENTIAL<span>{round(r_h) - round(r_a):+d}</span></div>
            <div class="elo-chip">ELO<span>{round(r_a)}</span></div>
          </div>
          <div class="prob-strip">
            <div class="prob-block {'prob-active' if pred_idx == 0 else ''}">
              <div class="prob-team">{home}</div>
              <div class="prob-pct">{probs[0]:.1%}</div>
              <div class="prob-lbl">Win</div>
            </div>
            <div class="draw-block prob-block {'prob-active' if pred_idx == 1 else ''}">
              <div class="prob-team">Draw</div>
              <div class="prob-pct">{probs[1]:.1%}</div>
              <div class="prob-lbl">Match nul</div>
            </div>
            <div class="prob-block {'prob-active' if pred_idx == 2 else ''}">
              <div class="prob-team">{away}</div>
              <div class="prob-pct">{probs[2]:.1%}</div>
              <div class="prob-lbl">Win</div>
            </div>
          </div>
          <div class="pred-label">Model Prediction</div>
          <div class="pred-value">{pred_display}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── PROBABILITY BAR CHART
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Outcome Probabilities</div>', unsafe_allow_html=True)

        labels = [f"{home} Win", "Draw", f"{away} Win"]
        values = [float(probs[0]), float(probs[1]), float(probs[2])]
        colors = ["#c9a84c", "#3d5a8a", "#e05c5c"]
        highlight = [1.0 if i == pred_idx else 0.65 for i in range(3)]
        bar_colors = [f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},{h})"
                      for c, h in zip(colors, highlight)]

        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=bar_colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            textfont=dict(family="Inter", size=15, color="#ffffff"),
            width=0.45,
        ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#8a9bbf"),
            yaxis=dict(tickformat=".0%", range=[0, max(values) * 1.25],
                       gridcolor="#1e2d47", zeroline=False),
            xaxis=dict(tickfont=dict(size=13, color="#ffffff")),
            margin=dict(t=20, b=10, l=10, r=10),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── FORM COMPARISON
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Recent Form (last 10 matches, recency-weighted)</div>',
                    unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric(f"{home} Win Rate", f"{h_form['form_w_rate']:.1%}")
            st.metric(f"{away} Win Rate", f"{a_form['form_w_rate']:.1%}")
        with fc2:
            st.metric(f"{home} Avg GD", f"{h_form['form_avg_gd']:+.2f}")
            st.metric(f"{away} Avg GD", f"{a_form['form_avg_gd']:+.2f}")
        with fc3:
            st.metric(f"{home} Avg GA", f"{h_form['form_avg_ga']:.2f}")
            st.metric(f"{away} Avg GA", f"{a_form['form_avg_ga']:.2f}")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── H2H NOTE
        st.markdown(f"""
        <div class="disclaimer">
          H2H encounters in dataset: {h2h_n} &nbsp;·&nbsp;
          Model trained on 44,000+ international matches (1960–2026) &nbsp;·&nbsp;
          Validation accuracy: 50–61% across WC 2010–2022<br>
          ⚠ Probabilities exclude injury/suspension data. Draw probability may be understated (~17% model vs ~25% WC historical base rate).
        </div>
        """, unsafe_allow_html=True)
