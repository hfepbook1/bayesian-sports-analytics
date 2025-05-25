# 🏀 Bayesian Sports Prediction App 🏆

[![Live Demo](https://img.shields.io/badge/Live%20Demo-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://sports-predictor.streamlit.app)
![Python 3.10](https://img.shields.io/badge/Python%203.10-blue?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)
![Last Updated](https://img.shields.io/badge/Last%20Updated-May%202025-brightgreen?style=flat)

*Predict NBA champions with Bayesian stats & optimize fantasy lineups with Monte Carlo simulation.*

---

## 🚀 Overview
**Bayesian Sports Prediction App** is an interactive Streamlit web app that tackles two hoops-analytics problems:

1. **NBA Championship Probabilities** – Uses a Bayesian Bradley-Terry model (PyMC) and thousands of simulated playoff brackets to estimate each team’s odds of winning the title.
2. **Daily Fantasy Lineup Optimization** – Runs Monte Carlo simulations of player performance, then picks the highest-EV lineup under the salary cap.

Everything runs in a friendly Streamlit UI—tweak parameters, watch the numbers change, and make smarter hoops decisions.

---

## ✨ Features
| Category | Highlights |
|---|---|
| 🏆 **Championship Predictor** | Posterior team strengths · Top-8 bracket simulation · Probability table & bar chart · “Roster-change” slider to test injuries/upgrades |
| 🤖 **DFS Optimizer** | Brute-force + MC lineup builder · Salary-cap aware · Histogram of total lineup points · Manual lineup-builder UI |
| 📊 **EDA Dashboards** | Interactive Plotly violin / histogram / scatter charts for team & player stats |
| 🎮 **Streamlit UX** | Multi-page layout · Instant re-simulation · Downloadable results |

---

## 🛠 Tech Stack
| Tool | Role |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core language (v3.10) |
| PyMC & Aesara | Bayesian inference & MCMC |
| ![NumPy](https://img.shields.io/badge/NumPy-777BB4?style=flat&logo=numpy&logoColor=white) / ![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=flat&logo=pandas&logoColor=white) | Data wrangling |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | App framework |
| ![Plotly](https://img.shields.io/badge/Plotly-239120?style=flat&logo=plotly&logoColor=white) | Interactive visualizations |

---

## ⚙️ Architecture
┌────────────────────────┐
│ Streamlit Front End │ ◄──── user interacts
└──────────┬─────────────┘
│ calls
┌──────────▼─────────────┐
│ Simulation Engine │ (PyMC: Bradley-Terry & MC)
└──────────┬─────────────┘
│ returns results
┌──────────▼─────────────┐
│ DFS Optimizer Module │ (salary-cap + MC totals)
└────────────────────────┘

---

## 📷 Screenshots
> Replace the image files after you capture them from the running app.

| Page | Preview |
|---|---|
| Championship Predictor | ![Championship](screenshots/champ_simulation.png) |
| DFS Optimizer | ![DFS](screenshots/dfs_optimizer.png) |

---

## ☁️ Live Demo
The app is deployed on **Streamlit Community Cloud**: **<https://sports-predictor.streamlit.app>**

---

## 💻 Local Setup

# 1. Clone
```bash
git clone https://github.com/yourusername/bayesian-sports-prediction-app.git
cd bayesian-sports-prediction-app
```

# 2. Create / activate a Python 3.10 env (optional)
```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```

# 3. Install deps
```bash
pip install -r requirements.txt
```

# 4. Run
```bash
streamlit run app.py
```
