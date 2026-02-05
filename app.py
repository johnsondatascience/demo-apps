"""
Viant Demo Project - Interactive Dashboard
Showcases ML capabilities relevant to ad tech: campaign performance, 
incrementality measurement, and supply forecasting.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_data import (
    generate_campaign_data, 
    generate_ab_test_data, 
    generate_supply_forecast_data
)

st.set_page_config(
    page_title="Ad Tech ML Demo - Michael Johnson",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🎯 Ad Tech ML Demo")
st.sidebar.markdown("**Michael Johnson**")
st.sidebar.markdown("Machine Learning Leader")
st.sidebar.markdown("---")

demo_options = [
    "🏠 Overview",
    "📈 Campaign Performance Prediction",
    "🔬 Bayesian Incrementality Measurement",
    "📅 Supply Forecasting"
]
selected_demo = st.sidebar.radio("Select Demo", demo_options)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About This Demo
Built to showcase ML skills relevant to Viant's ad tech platform:
- **XGBoost** for campaign optimization
- **PyMC Bayesian inference** for lift measurement
- **Holt-Winters** for inventory forecasting

[GitHub](https://github.com/johnsondatascience) | [LinkedIn](https://linkedin.com/in/data-arts-data-science)
""")


# ============== OVERVIEW ==============
if selected_demo == "🏠 Overview":
    st.title("Ad Tech Machine Learning Demo")
    st.markdown("### Built for Viant Technology Interview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📈 Campaign Performance
        **XGBoost model** predicting ROAS (Return on Ad Spend) based on:
        - Device type (CTV, Mobile, Desktop)
        - Vertical (Auto, Retail, Finance, etc.)
        - Budget, CPM, viewability metrics
        
        **Relevance:** Campaign optimization and performance modeling
        """)
    
    with col2:
        st.markdown("""
        #### 🔬 Incrementality Measurement
        **Bayesian A/B testing** framework measuring ad lift:
        - Posterior distributions of treatment effects
        - Credible intervals for business decisions
        - Probability of exceeding lift thresholds
        
        **Relevance:** Causal inference for ad attribution
        """)
    
    with col3:
        st.markdown("""
        #### 📅 Supply Forecasting
        **Prophet time series** model for ad inventory:
        - Daily/weekly/monthly seasonality
        - Holiday effects on CTV viewership
        - Confidence intervals for capacity planning
        
        **Relevance:** Inventory optimization and yield management
        """)
    
    st.markdown("---")
    st.markdown("### My Background in Ad Tech")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Pluto TV (Paramount) - 4 Years**
        - Led ML for ad-supported streaming (100M+ users)
        - Revenue forecasting for 30+ device types
        - Bayesian A/B testing framework (PyMC)
        - Churn prediction at scale (Ray on Databricks)
        """)
    
    with col2:
        st.markdown("""
        **Key Achievements**
        - Reduced forecast error (MAPE) by **5%**
        - Shortened SDLC by **20%** with Bayesian testing
        - **15% increase** in retention ROI
        - **3% lift** in marketing ROI
        """)
    
    st.markdown("---")
    st.info("👈 **Select a demo from the sidebar** to explore the technical implementations")


# ============== CAMPAIGN PERFORMANCE ==============
elif selected_demo == "📈 Campaign Performance Prediction":
    st.title("Campaign Performance Prediction")
    st.markdown("### XGBoost model predicting ROAS for CTV advertising campaigns")
    
    with st.spinner("Generating campaign data and training model..."):
        from models.campaign_performance import (
            CampaignPerformanceModel, 
            create_performance_visualizations
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            n_campaigns = st.slider("Number of campaigns", 100, 1000, 500)
        with col2:
            seed = st.number_input("Random seed", value=42)
        
        # Generate data and train
        df = generate_campaign_data(n_campaigns=n_campaigns, seed=seed)
        model = CampaignPerformanceModel()
        metrics, X_test, y_test, y_pred = model.train(df)
        
        # Metrics display
        st.markdown("### Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{metrics['r2']:.3f}")
        col2.metric("RMSE", f"{metrics['rmse']:.3f}")
        col3.metric("MAE", f"{metrics['mae']:.3f}")
        col4.metric("CV R² (mean)", f"{metrics['cv_scores'].mean():.3f}")
        
        # Visualizations
        figs = create_performance_visualizations(df, metrics, y_test, y_pred, model)
        
        st.markdown("### Feature Importance")
        st.plotly_chart(figs['feature_importance'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Actual vs Predicted")
            st.plotly_chart(figs['actual_vs_predicted'], use_container_width=True)
        with col2:
            st.markdown("### ROAS by Device Type")
            st.plotly_chart(figs['roas_by_device'], use_container_width=True)
        
        st.markdown("### ROAS by Vertical")
        st.plotly_chart(figs['roas_by_vertical'], use_container_width=True)
        
        st.markdown("### Budget vs ROAS")
        st.plotly_chart(figs['budget_vs_roas'], use_container_width=True)
        
        # Interactive prediction
        st.markdown("---")
        st.markdown("### Try a Prediction")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pred_vertical = st.selectbox("Vertical", df['vertical'].unique())
            pred_device = st.selectbox("Device Type", df['device_type'].unique())
        with col2:
            pred_budget = st.number_input("Budget ($)", value=50000, step=5000)
            pred_duration = st.slider("Duration (days)", 7, 90, 30)
        with col3:
            pred_cpm = st.number_input("CPM ($)", value=20.0, step=1.0)
            pred_vcr = st.slider("Video Completion Rate", 0.0, 1.0, 0.75)
        
        pred_viewability = st.slider("Viewability", 0.0, 1.0, 0.70)
        
        if st.button("Predict ROAS"):
            pred_df = pd.DataFrame([{
                'vertical': pred_vertical,
                'device_type': pred_device,
                'daypart': 'Primetime',
                'budget': pred_budget,
                'duration_days': pred_duration,
                'cpm': pred_cpm,
                'vcr': pred_vcr,
                'viewability': pred_viewability
            }])
            prediction = model.predict(pred_df)[0]
            
            st.success(f"**Predicted ROAS: {prediction:.2f}x**")
            st.markdown(f"Expected revenue: **${pred_budget * prediction:,.0f}** on ${pred_budget:,.0f} spend")


# ============== BAYESIAN INCREMENTALITY ==============
elif selected_demo == "🔬 Bayesian Incrementality Measurement":
    st.title("Bayesian Incrementality Measurement")
    st.markdown("### Causal inference for measuring ad campaign lift using PyMC")
    
    st.markdown("""
    This demo shows how to measure the **incremental impact** of ad exposure using 
    Bayesian statistics. Unlike frequentist A/B testing, this approach provides:
    - Full posterior distributions (not just point estimates)
    - Direct probability statements ("87% chance lift > 5%")
    - Better handling of small samples and uncertainty
    """)
    
    # Check if PyMC is available
    try:
        from models.bayesian_incrementality import (
            BayesianIncrementalityModel,
            create_incrementality_visualizations
        )
        pymc_available = True
    except (ImportError, ModuleNotFoundError) as e:
        pymc_available = False
        import_error = str(e)
    
    if not pymc_available:
        st.error(f"""
        **PyMC not available in this environment.**
        
        This demo requires PyMC with NumPy 2.0+. The current environment has a NumPy version conflict 
        with other packages (neuralprophet requires NumPy <2.0).
        
        **To run this demo**, create a separate environment with:
        ```
        pip install pymc arviz numpy>=2.0
        ```
        
        Error: `{import_error}`
        """)
        
        st.markdown("---")
        st.markdown("### Demo Preview (Static)")
        st.image("https://www.pymc.io/projects/docs/en/stable/_images/pymc_logo.png", width=200)
        st.markdown("""
        **What this demo would show:**
        - Posterior distribution of lift (treatment effect)
        - 95% credible intervals for decision-making
        - P(lift > 0%) and P(lift > 5%) for business decisions
        - Comparison of control vs treatment conversion rates
        
        **Code architecture:**
        - `BayesianIncrementalityModel` class using PyMC
        - Beta priors on conversion rates
        - Binomial likelihood for observed conversions
        - Derived lift = (p_treatment - p_control) / p_control
        """)
    else:
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            n_users = st.slider("Number of users", 1000, 50000, 10000)
        with col2:
            true_lift = st.slider("True lift (for simulation)", 0.0, 0.20, 0.05, 0.01)
        with col3:
            seed = st.number_input("Random seed", value=42, key="bayes_seed")
        
        if st.button("Run Bayesian Analysis", type="primary"):
            with st.spinner("Generating data and fitting Bayesian model... (this may take 30-60 seconds)"):
                # Generate data
                df = generate_ab_test_data(n_users=n_users, true_lift=true_lift, seed=seed)
                
                # Fit model
                model = BayesianIncrementalityModel()
                results = model.fit(df)
                
                # Display results
                st.markdown("### Results Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Estimated Lift", f"{results['lift_mean']:.1%}")
                col2.metric("True Lift (sim)", f"{true_lift:.1%}")
                col3.metric("P(Lift > 0)", f"{results['prob_positive_lift']:.0%}")
                col4.metric("P(Lift > 5%)", f"{results['prob_lift_gt_5pct']:.0%}")
                
                st.markdown(f"""
                **95% Credible Interval:** [{results['lift_95_ci'][0]:.1%}, {results['lift_95_ci'][1]:.1%}]
                """)
                
                # Visualizations
                figs = create_incrementality_visualizations(results)
                
                st.markdown("### Posterior Distribution of Lift")
                st.plotly_chart(figs['lift_distribution'], use_container_width=True)
                
                st.markdown("### Conversion Rate Distributions")
                st.plotly_chart(figs['cvr_comparison'], use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Threshold Probabilities")
                    st.plotly_chart(figs['threshold_probabilities'], use_container_width=True)
                with col2:
                    st.markdown("### Credible Intervals")
                    st.plotly_chart(figs['credible_intervals'], use_container_width=True)
                
                # Business interpretation
                st.markdown("---")
                st.markdown("### Business Interpretation")
                
                if results['prob_positive_lift'] > 0.95:
                    st.success(f"""
                    ✅ **Strong evidence of positive lift.** There is a {results['prob_positive_lift']:.0%} probability 
                    that the campaign had a positive effect on conversions.
                    """)
                elif results['prob_positive_lift'] > 0.80:
                    st.warning(f"""
                    ⚠️ **Moderate evidence of positive lift.** There is a {results['prob_positive_lift']:.0%} probability 
                    that the campaign had a positive effect. Consider extending the test for more certainty.
                    """)
                else:
                    st.error(f"""
                    ❌ **Insufficient evidence of lift.** Only {results['prob_positive_lift']:.0%} probability 
                    of positive effect. The campaign may not be driving incremental conversions.
                    """)
        else:
            st.info("👆 Click 'Run Bayesian Analysis' to fit the model")


# ============== SUPPLY FORECASTING ==============
elif selected_demo == "📅 Supply Forecasting":
    st.title("Ad Supply Forecasting")
    st.markdown("### Time series model for ad inventory prediction")
    
    st.markdown("""
    Accurate supply forecasting is critical for ad tech platforms to:
    - Optimize **fill rates** (maximize revenue from available inventory)
    - Set appropriate **floor prices** (CPM optimization)
    - Plan **capacity** for seasonal spikes (Q4, holidays)
    """)
    
    with st.spinner("Generating supply data and training forecast model..."):
        from models.supply_forecasting import (
            SupplyForecastModel,
            create_forecast_visualizations
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            n_days = st.slider("Historical days", 180, 730, 365)
        with col2:
            holdout_days = st.slider("Holdout days for validation", 14, 60, 30)
        
        # Generate data and train
        df = generate_supply_forecast_data(n_days=n_days)
        model = SupplyForecastModel()
        metrics, test_results = model.train(df, holdout_days=holdout_days)
        
        # Metrics
        st.markdown("### Forecast Accuracy")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAPE", f"{metrics['mape']:.2f}%")
        col2.metric("MAE", f"{metrics['mae']/1e6:.1f}M impressions")
        col3.metric("95% CI Coverage", f"{metrics['coverage']:.1f}%")
        col4.metric("Train/Test", f"{metrics['train_size']}/{metrics['test_size']} days")
        
        # Visualizations
        figs = create_forecast_visualizations(df, model.forecast, metrics, model)
        
        st.markdown("### Forecast vs Actuals")
        st.plotly_chart(figs['forecast'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Weekly Pattern")
            st.plotly_chart(figs['weekly_seasonality'], use_container_width=True)
        with col2:
            st.markdown("### Monthly Pattern")
            st.plotly_chart(figs['monthly_seasonality'], use_container_width=True)
        
        st.markdown("### Fill Rate & CPM Dynamics")
        st.plotly_chart(figs['economics'], use_container_width=True)
        
        st.markdown("### Forecast Error Distribution")
        st.plotly_chart(figs['residuals'], use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.markdown("### Key Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Seasonality Patterns Detected:**
            - 📅 **Weekly:** Weekend viewing ~20% higher (CTV behavior)
            - 📆 **Monthly:** Q4 inventory peaks (holiday streaming)
            - 🎄 **Holiday:** 30% surge during Thanksgiving-Christmas
            """)
        with col2:
            st.markdown("""
            **Business Applications:**
            - Set dynamic floor prices based on predicted demand
            - Alert sales team when forecast shows undersold inventory
            - Plan infrastructure capacity for peak periods
            """)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("*Demo built for Viant Technology interview*")
st.sidebar.markdown("*February 2026*")
