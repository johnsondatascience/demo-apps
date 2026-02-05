# Ad Tech ML Demo - Viant Interview Showcase

Interactive demonstrations of machine learning capabilities relevant to Viant's programmatic advertising platform.

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📊 Demos Included

### 1. Campaign Performance Prediction
**Technology:** XGBoost, scikit-learn

Predicts ROAS (Return on Ad Spend) for CTV advertising campaigns based on:
- Device type (CTV, Mobile, Desktop, Tablet)
- Vertical (Automotive, Retail, Finance, etc.)
- Budget, duration, CPM
- Video completion rate, viewability

**Viant Relevance:** Campaign optimization, performance modeling for advertisers

### 2. Bayesian Incrementality Measurement
**Technology:** PyMC, ArviZ

Measures the causal lift from ad exposure using Bayesian inference:
- Posterior distributions of treatment effects
- Credible intervals for decision-making
- Direct probability statements (e.g., "92% chance lift > 5%")

**Viant Relevance:** Attribution, incremental value measurement for advertisers

### 3. Supply Forecasting
**Technology:** Holt-Winters (statsmodels), Plotly

Forecasts available ad inventory with time series modeling:
- Weekly seasonality (weekend CTV viewing patterns)
- Monthly/yearly trends (Q4 holiday peaks)
- Confidence intervals for capacity planning

**Viant Relevance:** Inventory optimization, yield management, pricing decisions

---

## 🎯 Talking Points for Interview

### On Campaign Performance Modeling
> "At Pluto TV, I owned revenue forecasting for 30+ device types. This demo shows how I approach campaign-level prediction—using XGBoost to identify which features drive ROAS. The CTV-heavy distribution reflects real viewing patterns I observed."

### On Bayesian Incrementality
> "I built the Bayesian A/B testing framework at Pluto TV using PyMC. The key advantage over frequentist methods is that we can make direct probability statements—'There's an 87% chance this feature improved retention'—which maps directly to business risk tolerance."

### On Supply Forecasting
> "My forecasts at Pluto TV were the primary input for executive financial planning. This Prophet model captures the same seasonality patterns I dealt with—weekend viewing spikes, Q4 holiday surges. A 5% MAPE improvement directly impacts revenue planning accuracy."

### On Scale
> "The synthetic data here is small for demo purposes, but at Pluto TV I worked with 100M+ users and 50TB+ monthly data processing on Databricks. The modeling approaches are identical—the infrastructure challenge is parallelization (Ray, PySpark)."

---

## 📁 Project Structure

```
demo_project/
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   ├── __init__.py
│   └── synthetic_data.py       # Data generators
└── models/
    ├── __init__.py
    ├── campaign_performance.py # XGBoost ROAS model
    ├── bayesian_incrementality.py  # PyMC lift measurement
    └── supply_forecasting.py   # Prophet forecasting
```

---

## 🔧 Technical Notes

### Why These Technologies?
| Tool | Why | Viant Relevance |
|------|-----|-----------------|
| **XGBoost** | Industry standard for tabular ML, interpretable | Campaign optimization at scale |
| **PyMC** | Gold standard for Bayesian inference | Incrementality measurement |
| **Holt-Winters** | Handles seasonality well, NumPy 2.0 compatible | Inventory forecasting |
| **Streamlit** | Fast to prototype, interactive | Stakeholder demos |

### Running Individual Models
```python
# Campaign Performance
from models.campaign_performance import run_campaign_demo
model, df, metrics, y_test, y_pred = run_campaign_demo()

# Bayesian Incrementality
from models.bayesian_incrementality import run_incrementality_demo
model, df, results = run_incrementality_demo(true_lift=0.05)

# Supply Forecasting
from models.supply_forecasting import run_forecast_demo
model, df, metrics = run_forecast_demo()
```

---

## 📞 Contact

**Michael Johnson**  
Machine Learning Leader | Causal Inference & Forecasting Specialist

- 📧 mchl.dvd.jhnsn@gmail.com
- 📱 (949) 340-4330
- 🔗 [LinkedIn](https://www.linkedin.com/in/data-arts-data-science)
- 📍 Orange County, CA

---

*Built for Viant Technology - Manager, Machine Learning position*  
*February 2026*
