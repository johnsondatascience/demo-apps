"""
Bayesian Incrementality Measurement
Demonstrates causal inference for measuring ad campaign lift using PyMC.
This is a core capability for ad tech measurement teams.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class BayesianIncrementalityModel:
    """
    Bayesian A/B test analysis for measuring incremental lift from ad exposure.
    Uses PyMC for posterior inference on conversion rate differences.
    """
    
    def __init__(self):
        self.trace = None
        self.model = None
        self.results = None
        
    def fit(self, df: pd.DataFrame) -> dict:
        """
        Fit Bayesian model to A/B test data.
        Estimates posterior distribution of lift (treatment effect).
        """
        # Aggregate data
        control = df[df['treatment'] == 0]
        treatment = df[df['treatment'] == 1]
        
        n_control = len(control)
        n_treatment = len(treatment)
        conversions_control = control['converted'].sum()
        conversions_treatment = treatment['converted'].sum()
        
        # Build PyMC model
        with pm.Model() as self.model:
            # Priors for conversion rates (weakly informative Beta)
            p_control = pm.Beta('p_control', alpha=2, beta=20)
            p_treatment = pm.Beta('p_treatment', alpha=2, beta=20)
            
            # Likelihood
            obs_control = pm.Binomial('obs_control', n=n_control, p=p_control, 
                                     observed=conversions_control)
            obs_treatment = pm.Binomial('obs_treatment', n=n_treatment, p=p_treatment,
                                       observed=conversions_treatment)
            
            # Derived quantities
            lift = pm.Deterministic('lift', (p_treatment - p_control) / p_control)
            lift_absolute = pm.Deterministic('lift_absolute', p_treatment - p_control)
            
            # Sample posterior
            self.trace = pm.sample(2000, tune=1000, cores=1, random_seed=42, 
                                  progressbar=False)
        
        # Extract results
        posterior = self.trace.posterior
        
        self.results = {
            'n_control': n_control,
            'n_treatment': n_treatment,
            'conversions_control': conversions_control,
            'conversions_treatment': conversions_treatment,
            'observed_cvr_control': conversions_control / n_control,
            'observed_cvr_treatment': conversions_treatment / n_treatment,
            'posterior_p_control': posterior['p_control'].values.flatten(),
            'posterior_p_treatment': posterior['p_treatment'].values.flatten(),
            'posterior_lift': posterior['lift'].values.flatten(),
            'posterior_lift_absolute': posterior['lift_absolute'].values.flatten()
        }
        
        # Calculate credible intervals and probabilities
        lift_samples = self.results['posterior_lift']
        self.results['lift_mean'] = lift_samples.mean()
        self.results['lift_median'] = np.median(lift_samples)
        self.results['lift_95_ci'] = (np.percentile(lift_samples, 2.5), 
                                      np.percentile(lift_samples, 97.5))
        self.results['prob_positive_lift'] = (lift_samples > 0).mean()
        self.results['prob_lift_gt_5pct'] = (lift_samples > 0.05).mean()
        
        return self.results
    
    def get_summary(self) -> str:
        """Generate human-readable summary of results."""
        r = self.results
        summary = f"""
Bayesian Incrementality Analysis Results
========================================

Sample Sizes:
  Control: {r['n_control']:,} users ({r['conversions_control']:,} conversions)
  Treatment: {r['n_treatment']:,} users ({r['conversions_treatment']:,} conversions)

Observed Conversion Rates:
  Control: {r['observed_cvr_control']:.2%}
  Treatment: {r['observed_cvr_treatment']:.2%}

Posterior Estimates (Relative Lift):
  Mean Lift: {r['lift_mean']:.1%}
  Median Lift: {r['lift_median']:.1%}
  95% Credible Interval: [{r['lift_95_ci'][0]:.1%}, {r['lift_95_ci'][1]:.1%}]

Business Decision Metrics:
  P(Lift > 0%): {r['prob_positive_lift']:.1%}
  P(Lift > 5%): {r['prob_lift_gt_5pct']:.1%}
"""
        return summary


def create_incrementality_visualizations(results: dict) -> dict:
    """Create interactive visualizations for incrementality analysis."""
    
    figs = {}
    
    # 1. Posterior distribution of lift
    fig_lift = go.Figure()
    fig_lift.add_trace(go.Histogram(
        x=results['posterior_lift'] * 100,
        nbinsx=50,
        name='Posterior Lift',
        marker_color='steelblue',
        opacity=0.7
    ))
    fig_lift.add_vline(x=0, line_dash="dash", line_color="red", 
                      annotation_text="No Effect")
    fig_lift.add_vline(x=results['lift_mean'] * 100, line_dash="solid", 
                      line_color="green", annotation_text=f"Mean: {results['lift_mean']:.1%}")
    fig_lift.update_layout(
        title='Posterior Distribution of Relative Lift',
        xaxis_title='Lift (%)',
        yaxis_title='Frequency',
        showlegend=False
    )
    figs['lift_distribution'] = fig_lift
    
    # 2. Conversion rate comparison
    fig_cvr = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Control Group', 'Treatment Group'))
    
    fig_cvr.add_trace(
        go.Histogram(x=results['posterior_p_control'] * 100, nbinsx=40,
                    marker_color='coral', name='Control'),
        row=1, col=1
    )
    fig_cvr.add_trace(
        go.Histogram(x=results['posterior_p_treatment'] * 100, nbinsx=40,
                    marker_color='mediumseagreen', name='Treatment'),
        row=1, col=2
    )
    fig_cvr.update_layout(title='Posterior Distributions of Conversion Rates')
    fig_cvr.update_xaxes(title_text='Conversion Rate (%)', row=1, col=1)
    fig_cvr.update_xaxes(title_text='Conversion Rate (%)', row=1, col=2)
    figs['cvr_comparison'] = fig_cvr
    
    # 3. Probability of exceeding thresholds
    thresholds = [0, 0.02, 0.05, 0.10, 0.15, 0.20]
    probabilities = [(results['posterior_lift'] > t).mean() for t in thresholds]
    
    fig_prob = px.bar(
        x=[f'>{t:.0%}' for t in thresholds],
        y=[p * 100 for p in probabilities],
        labels={'x': 'Lift Threshold', 'y': 'Probability (%)'},
        title='Probability of Exceeding Various Lift Thresholds'
    )
    fig_prob.add_hline(y=95, line_dash="dash", line_color="green",
                      annotation_text="95% Confidence")
    figs['threshold_probabilities'] = fig_prob
    
    # 4. Credible interval visualization
    ci_levels = [50, 80, 90, 95, 99]
    ci_data = []
    for level in ci_levels:
        lower = np.percentile(results['posterior_lift'], (100 - level) / 2)
        upper = np.percentile(results['posterior_lift'], 100 - (100 - level) / 2)
        ci_data.append({'level': f'{level}%', 'lower': lower * 100, 'upper': upper * 100})
    
    ci_df = pd.DataFrame(ci_data)
    
    fig_ci = go.Figure()
    for i, row in ci_df.iterrows():
        fig_ci.add_trace(go.Scatter(
            x=[row['lower'], row['upper']],
            y=[row['level'], row['level']],
            mode='lines+markers',
            name=row['level'],
            line=dict(width=8 - i),
            marker=dict(size=10)
        ))
    fig_ci.add_vline(x=0, line_dash="dash", line_color="red")
    fig_ci.update_layout(
        title='Credible Intervals for Lift Estimate',
        xaxis_title='Lift (%)',
        yaxis_title='Credible Interval Level'
    )
    figs['credible_intervals'] = fig_ci
    
    return figs


def run_incrementality_demo(true_lift: float = 0.05):
    """Run the full incrementality measurement demo."""
    from data.synthetic_data import generate_ab_test_data
    
    # Generate data with known true lift
    df = generate_ab_test_data(n_users=10000, true_lift=true_lift)
    
    # Fit model
    model = BayesianIncrementalityModel()
    results = model.fit(df)
    
    print(model.get_summary())
    print(f"\n(True lift used in simulation: {true_lift:.1%})")
    
    return model, df, results


if __name__ == "__main__":
    run_incrementality_demo()
