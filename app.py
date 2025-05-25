import streamlit as st
import numpy as np
import pandas as pd
import math
import itertools
import plotly.express as px
try:
    import pymc as pm
except ImportError:
    import pymc3 as pm
import aesara.tensor as at

st.set_page_config(page_title="Bayesian Sports Prediction", page_icon="🏀", layout="wide")

# ── Simulation Helpers ──
def simulate_series(teamA, teamB, ratings):
    rA, rB = ratings[teamA], ratings[teamB]
    pA = 1.0 / (1.0 + math.exp(rB - rA))
    winsA = np.random.binomial(7, pA)
    return teamA if winsA >= 4 else teamB

def simulate_championship(ratings, n_sim=5000, rating_sd=0.5):
    teams = list(ratings.keys())
    counts = {t: 0 for t in teams}
    pairs = [(teams[0],teams[7]),(teams[1],teams[6]),(teams[2],teams[5]),(teams[3],teams[4])]
    for _ in range(n_sim):
        sim_r = {t: np.random.normal(ratings[t], rating_sd) for t in teams}
        q = [simulate_series(a,b,sim_r) for a,b in pairs]
        s1 = simulate_series(q[0],q[3],sim_r)
        s2 = simulate_series(q[1],q[2],sim_r)
        champ = simulate_series(s1,s2,sim_r)
        counts[champ] += 1
    return {t: counts[t]/n_sim for t in teams}

# ── Sidebar Uploads ──
st.sidebar.header("Optional Data Uploads")
nba_upload = st.sidebar.file_uploader("Upload NBA Games CSV", type=["csv"])
dfs_upload = st.sidebar.file_uploader("Upload DFS Players CSV", type=["csv"])

# ── NBA Data Prep ──
if nba_upload:
    raw = pd.read_csv(nba_upload, parse_dates=["date"])
    raw = raw.dropna(subset=["score1","score2"])
    raw["winner"] = np.where(raw.score1 > raw.score2, raw.team1, raw.team2)
    raw["loser" ] = np.where(raw.score1 > raw.score2, raw.team2, raw.team1)
    team_names = list(pd.unique(raw[["team1","team2"]].values.ravel()))
    idx = {t:i for i,t in enumerate(team_names)}
    i_idx = raw.winner.map(idx).values
    j_idx = raw.loser .map(idx).values
    outcomes = np.ones_like(i_idx)
else:
    team_names = [
        "Bucks","Celtics","76ers","Cavs","Knicks","Nets","Heat","Hawks",
        "Nuggets","Suns","Grizzlies","Warriors","Lakers","Clippers","Kings","Timberwolves"
    ]
    idx = {t:i for i,t in enumerate(team_names)}
    np.random.seed(42)
    true_s = np.random.normal(0,1,len(team_names))
    games = []
    for i in range(len(team_names)):
        for j in range(i+1,len(team_names)):
            p = 1/(1+math.exp(-(true_s[i]-true_s[j])))
            o = 1 if np.random.rand()<p else 0
            games.append({"team_i":i,"team_j":j,"team_i_win":o})
    df_g = pd.DataFrame(games)
    i_idx, j_idx = df_g.team_i.values, df_g.team_j.values
    outcomes = df_g.team_i_win.values

# ── DFS Data Prep ──
if dfs_upload:
    players_df = pd.read_csv(dfs_upload)
else:
    positions = ["PG","SG","SF","PF","C"]
    np.random.seed(24)
    tmp = []
    for p in positions:
        for i in range(1,11):
            sal = np.random.randint(5000,12001)
            fac = {"PG":0.9,"SG":0.9,"SF":1.0,"PF":1.0,"C":1.1}[p] + np.random.normal(0,0.1)
            fac = max(fac,0.1)
            proj = (sal/1000)*5*fac
            tmp.append({"Name":f"{p}_{i}","Position":p,"Salary":sal,"Projection":proj})
    players_df = pd.DataFrame(tmp)

# Normalize DFS columns
if "Player" in players_df.columns and "Name" not in players_df.columns:
    players_df = players_df.rename(columns={"Player":"Name"})
if "ProjPoints" in players_df.columns and "Projection" not in players_df.columns:
    players_df = players_df.rename(columns={"ProjPoints":"Projection"})
players_df = players_df[["Name","Position","Salary","Projection"]]

# ── Bradley–Terry Fit & Playoff Sim ──
if "team_data" not in st.session_state:
    n = len(team_names)
    with pm.Model() as model:
        strength = pm.Normal("strength", 0, 1, shape=n)
        pm.Data("i_idx", i_idx)
        pm.Data("j_idx", j_idx)
        p = pm.math.sigmoid(strength[i_idx] - strength[j_idx])
        pm.Bernoulli("obs", p=p, observed=outcomes)
        trace = pm.sample(1000, tune=1000, chains=1,
                          random_seed=42, progressbar=False,
                          return_inferencedata=True)
    samples = trace.posterior["strength"].values.reshape(-1, n)
    mean_s = samples.mean(axis=0)

    # pick top 8 teams by mean strength
    order = np.argsort(mean_s)[::-1]
    top8 = [team_names[i] for i in order[:8]]
    base_ratings = {t: mean_s[team_names.index(t)] for t in top8}
    champ_probs = simulate_championship(base_ratings)

    summary = []
    for i, t in enumerate(team_names):
        lo, hi = np.percentile(samples[:, i], [2.5, 97.5])
        summary.append({
            "Team": t,
            "Strength (mean)": f"{mean_s[i]:.2f}",
            "95% CI": f"[{lo:.2f},{hi:.2f}]",
            "Champ. Prob": (f"{champ_probs[t]*100:.1f}%" if t in champ_probs else "—")
        })
    summary_df = pd.DataFrame(summary)

    samp_df = pd.DataFrame(samples, columns=team_names)
    long_df = samp_df.melt(var_name="Team", value_name="Strength")
    long_df["Team"] = pd.Categorical(long_df["Team"], categories=team_names, ordered=True)

    st.session_state.team_data = {
        "team_names": team_names,
        "mean_s": mean_s,
        "top8": top8,
        "base_ratings": base_ratings,
        "champ_probs": champ_probs,
        "summary_df": summary_df,
        "long_df": long_df
    }

st.session_state.players = players_df

# ── Streamlit Tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "NBA Champion Predictor",
    "Daily Fantasy Optimizer",
    "NBA Dataset EDA",
    "DFS Dataset EDA",
    "App Documentation"
])

# Tab 1: NBA Champion Predictor
with tab1:
    td = st.session_state.team_data
    st.header("NBA Champion Predictor")

    # Baseline probabilities table
    df_base = pd.DataFrame({
        "Team": td["champ_probs"].keys(),
        "Probability": td["champ_probs"].values()
    }).sort_values("Probability", ascending=False)
    df_base["Probability"] = df_base["Probability"].map(lambda x: f"{x*100:.1f}%")
    st.subheader("Baseline Championship Probabilities (Top 8)")
    st.table(df_base)

    # Simulate Roster Change
    st.subheader("Simulate Roster Change")
    st.markdown("""
Pick one of the **Top 8** teams below and slide to simulate an injury (negative) or upgrade (positive).
The championship probabilities will be recalculated for that same bracket.
""")
    teams_for_slider = td.get("top8", list(td["champ_probs"].keys()))
    team_sel = st.selectbox("Team to adjust:", teams_for_slider)
    adjustment = st.slider("Strength adjustment", -3.0, 3.0, 0.0, 0.1)

    if adjustment != 0.0:
        # build fallback ratings list if needed
        ratings_list = td.get("mean_s", np.zeros(len(td["team_names"])))
        fallback_ratings = {
            t: ratings_list[td["team_names"].index(t)]
            for t in teams_for_slider
        }
        base = td.get("base_ratings", fallback_ratings)
        mod = base.copy()
        mod[team_sel] += adjustment

        new_probs = simulate_championship(mod)
        df_new = pd.DataFrame({
            "Team": new_probs.keys(),
            "Probability": new_probs.values()
        }).sort_values("Probability", ascending=False)
        df_new["Probability"] = df_new["Probability"].map(lambda x: f"{x*100:.1f}%")
        st.markdown(f"**With {team_sel} adjusted by {adjustment:+.1f}:**")
        st.table(df_new)
    else:
        st.write("*(No adjustment applied. Move the slider to simulate.)*")

# Tab 2: Daily Fantasy Optimizer
with tab2:
    dfp = st.session_state.players
    st.header("Daily Fantasy Optimizer")

    st.subheader("Automatically Optimized Lineup")
    cap = st.slider("Salary Cap", 30000, 60000, 50000, 1000)
    pos = ["PG","SG","SF","PF","C"]
    groups = {p: dfp[dfp.Position==p].to_dict('records') for p in pos}
    best, best_pts = None, -np.inf
    for combo in itertools.product(*[groups[p] for p in pos]):
        s = sum(p["Salary"] for p in combo)
        if s <= cap:
            pts = sum(p["Projection"] for p in combo)
            if pts > best_pts:
                best_pts, best = pts, combo
    if best:
        auto_df = pd.DataFrame(best)[["Name","Position","Salary","Projection"]]
        st.table(auto_df.style.format({"Salary":"${:,.0f}","Projection":"{:.1f}"}))
        st.write(f"Total Salary: ${sum(p['Salary'] for p in best):,}")
        st.write(f"Total Projected Points: {sum(p['Projection'] for p in best):.1f}")
    else:
        st.write("No valid lineup under that cap.")

    st.markdown("---")
    st.subheader("Build Your Own Lineup")
    picks = {p: st.selectbox(f"{p}:", dfp[dfp.Position==p]["Name"], key=p) for p in pos}
    custom = dfp[dfp.Name.isin(picks.values())]
    ts, tp = custom.Salary.sum(), custom.Projection.sum()
    st.markdown(f"**Total Salary:** ${ts:,}   **Projected Points:** {tp:.1f}")

    sims = 1000
    m = custom.Projection.values
    sd = 0.15 * m
    mat = np.random.normal(m.reshape(-1,1), sd.reshape(-1,1), (len(m), sims))
    mat[mat < 0] = 0
    totals = mat.sum(axis=0)
    fig = px.histogram(
        x=totals, nbins=20,
        title="Distribution of Simulated Total Fantasy Points",
        labels={"x":"Total Points","y":"Frequency"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
Each player's fantasy score is modeled as Normal(projection, 15% sd).  
Monte Carlo (1,000 sims) → histogram above.
""")

# Tab 3: NBA Dataset EDA
with tab3:
    td = st.session_state.team_data
    st.header("NBA Dataset EDA")
    chosen = st.multiselect("Teams:", td["team_names"], default=td["team_names"])
    if chosen:
        dfv = td["long_df"]
        dfv = dfv[dfv.Team.isin(chosen)]
        figv = px.violin(
            dfv, x="Team", y="Strength", color="Team",
            box=True, points=False,
            title="Posterior Distribution of Team Strengths"
        )
        st.plotly_chart(figv, use_container_width=True)
    st.subheader("Champion Probabilities (Top 8)")
    cp8 = pd.DataFrame({
        "Team": list(td["champ_probs"].keys()),
        "Probability": list(td["champ_probs"].values())
    }).sort_values("Probability", ascending=False)
    cp8["Probability"] = cp8["Probability"].map(lambda x: f"{x*100:.1f}%")
    st.table(cp8)

# Tab 4: DFS Dataset EDA
with tab4:
    dfp = st.session_state.players
    st.header("DFS Dataset EDA")
    pos = ["PG","SG","SF","PF","C"]
    pf = st.selectbox("Salary Filter:", ["All"] + pos)
    sal_data = dfp.Salary if pf=="All" else dfp[dfp.Position==pf].Salary
    st.plotly_chart(px.histogram(sal_data, nbins=20, title=f"Salary Distribution ({pf})"), use_container_width=True)
    pf2 = st.selectbox("Projection Filter:", ["All"] + pos, key="pf2")
    pts_data = dfp.Projection if pf2=="All" else dfp[dfp.Position==pf2].Projection
    st.plotly_chart(px.histogram(pts_data, nbins=20, title=f"Projection Distribution ({pf2})"), use_container_width=True)
    chos = st.multiselect("Scatter Positions:", pos, default=pos)
    scdf = dfp[dfp.Position.isin(chos)]
    st.plotly_chart(px.scatter(scdf, x="Salary", y="Projection", color="Position", title="Salary vs Projection"), use_container_width=True)
    st.plotly_chart(px.box(dfp, x="Position", y="Projection", color="Position", title="Projection by Position"), use_container_width=True)

# Tab 5: App Documentation
with tab5:
    st.markdown("""
# App Documentation

**NBA Champion Predictor**  
- Data: upload real game logs or fall back to synthetic  
- Model: Bayesian Bradley–Terry (PyMC)  
- Bracket: select Top 8 by inferred strength & simulate playoff  
- Roster Change: slider adjusts one team's rating, probabilities update

**Daily Fantasy Optimizer**  
- Automatic lineup optimization under salary cap  
- Manual lineup builder + Monte Carlo histogram of total points

**EDA Tabs**  
- Interactive Plotly charts for team-strength posteriors & DFS distributions

**Data Upload**  
- Use the sidebar to override synthetic data with your Kaggle CSVs

**Packages**  
`streamlit`, `numpy`, `pandas`, `pymc`/`pymc3`, `aesara`, `plotly`, `itertools`
""")
