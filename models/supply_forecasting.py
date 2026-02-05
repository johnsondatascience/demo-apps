"""
Ad Supply Forecasting Model
Forecasts available ad inventory using time series modeling.
Critical for ad tech platforms to optimize fill rates and pricing.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


class SupplyForecastModel:
    """
    Exponential Smoothing forecasting model for ad inventory prediction.
    Captures weekly and seasonal patterns using Holt-Winters method.
    Compatible with NumPy 2.0+.
    """
    
    def __init__(self):
        self.model = None
        self.train_df = None
        self.forecast = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'available_impressions') -> pd.DataFrame:
        """Prepare data for forecasting (requires 'ds' and 'y' columns)."""
        forecast_df = df[['date', target_col]].copy()
        forecast_df.columns = ['ds', 'y']
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        forecast_df = forecast_df.set_index('ds')
        return forecast_df
    
    def train(self, df: pd.DataFrame, target_col: str = 'available_impressions',
              holdout_days: int = 30) -> dict:
        """
        Train Holt-Winters model with train/test split for validation.
        Returns performance metrics on holdout set.
        """
        forecast_df = self.prepare_data(df, target_col)
        
        # Split into train and test
        cutoff_date = forecast_df.index.max() - pd.Timedelta(days=holdout_days)
        train = forecast_df[forecast_df.index <= cutoff_date]
        test = forecast_df[forecast_df.index > cutoff_date]
        
        self.train_df = train
        
        # Configure Holt-Winters model with weekly seasonality (7 days)
        self.model = ExponentialSmoothing(
            train['y'],
            seasonal_periods=7,
            trend='add',
            seasonal='mul',
            damped_trend=True
        )
        
        # Fit model
        fitted_model = self.model.fit(optimized=True)
        
        # Generate forecast for test period + 30 days
        forecast_periods = holdout_days + 30
        yhat = fitted_model.forecast(forecast_periods)
        
        # Build forecast dataframe
        future_dates = pd.date_range(
            start=train.index.max() + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        # Calculate confidence intervals (approximate using residual std)
        residuals = fitted_model.fittedvalues - train['y']
        std_resid = residuals.std()
        
        self.forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': yhat.values,
            'yhat_lower': yhat.values - 1.96 * std_resid,
            'yhat_upper': yhat.values + 1.96 * std_resid
        })
        
        # Add historical fitted values
        historical = pd.DataFrame({
            'ds': train.index,
            'yhat': fitted_model.fittedvalues.values,
            'yhat_lower': fitted_model.fittedvalues.values - 1.96 * std_resid,
            'yhat_upper': fitted_model.fittedvalues.values + 1.96 * std_resid
        })
        self.forecast = pd.concat([historical, self.forecast], ignore_index=True)
        
        # Calculate metrics on holdout
        test_dates = test.index
        test_forecast = self.forecast[self.forecast['ds'].isin(test_dates)].copy()
        test_reset = test.reset_index()
        test_reset.columns = ['ds', 'y']
        merged = test_reset.merge(test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
        
        metrics = {
            'mae': mean_absolute_error(merged['y'], merged['yhat']),
            'mape': mean_absolute_percentage_error(merged['y'], merged['yhat']) * 100,
            'coverage': ((merged['y'] >= merged['yhat_lower']) & 
                        (merged['y'] <= merged['yhat_upper'])).mean() * 100,
            'holdout_days': holdout_days,
            'train_size': len(train),
            'test_size': len(test)
        }
        
        return metrics, merged
    
    def forecast_future(self, periods: int = 30) -> pd.DataFrame:
        """Generate forecast for future periods."""
        return self.forecast.tail(periods)
    
    def get_components(self) -> pd.DataFrame:
        """Extract forecast components from the model."""
        return self.forecast


def create_forecast_visualizations(df: pd.DataFrame, forecast: pd.DataFrame,
                                   metrics: dict, model: SupplyForecastModel) -> dict:
    """Create interactive visualizations for supply forecasting."""
    
    figs = {}
    
    # 1. Forecast with actuals and confidence interval
    fig_forecast = go.Figure()
    
    # Actuals
    fig_forecast.add_trace(go.Scatter(
        x=df['date'],
        y=df['available_impressions'],
        mode='lines',
        name='Actual',
        line=dict(color='steelblue')
    ))
    
    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='coral', dash='dash')
    ))
    
    # Confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(255,127,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
    
    fig_forecast.update_layout(
        title=f'Ad Inventory Forecast (MAPE: {metrics["mape"]:.1f}%)',
        xaxis_title='Date',
        yaxis_title='Available Impressions',
        hovermode='x unified'
    )
    figs['forecast'] = fig_forecast
    
    # 2. Weekly seasonality pattern
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly = df.groupby('day_of_week')['available_impressions'].mean().reset_index()
    weekly['day_name'] = weekly['day_of_week'].map(dict(enumerate(days)))
    
    fig_weekly = px.bar(
        weekly, x='day_name', y='available_impressions',
        title='Weekly Seasonality Pattern (Avg Impressions by Day)',
        labels={'available_impressions': 'Avg Impressions', 'day_name': 'Day of Week'}
    )
    figs['weekly_seasonality'] = fig_weekly
    
    # 3. Monthly trends
    monthly = df.groupby('month')['available_impressions'].mean().reset_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly['month_name'] = monthly['month'].map(lambda x: month_names[x-1])
    
    fig_monthly = px.bar(
        monthly, x='month_name', y='available_impressions',
        title='Monthly Seasonality Pattern (Avg Impressions by Month)',
        labels={'available_impressions': 'Avg Impressions', 'month_name': 'Month'},
        color='available_impressions',
        color_continuous_scale='RdYlGn'
    )
    figs['monthly_seasonality'] = fig_monthly
    
    # 4. Fill rate vs CPM relationship
    fig_economics = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_economics.add_trace(
        go.Scatter(x=df['date'], y=df['fill_rate'] * 100, name='Fill Rate (%)',
                  line=dict(color='mediumseagreen')),
        secondary_y=False
    )
    fig_economics.add_trace(
        go.Scatter(x=df['date'], y=df['avg_cpm'], name='Avg CPM ($)',
                  line=dict(color='coral')),
        secondary_y=True
    )
    
    fig_economics.update_layout(title='Fill Rate vs CPM Over Time')
    fig_economics.update_yaxes(title_text='Fill Rate (%)', secondary_y=False)
    fig_economics.update_yaxes(title_text='Avg CPM ($)', secondary_y=True)
    figs['economics'] = fig_economics
    
    # 5. Forecast accuracy (residuals)
    merged = df.merge(forecast[['ds', 'yhat']], left_on='date', right_on='ds', how='inner')
    merged['residual'] = merged['available_impressions'] - merged['yhat']
    merged['residual_pct'] = merged['residual'] / merged['available_impressions'] * 100
    
    fig_residuals = px.histogram(
        merged, x='residual_pct', nbins=50,
        title='Forecast Error Distribution',
        labels={'residual_pct': 'Error (%)'}
    )
    fig_residuals.add_vline(x=0, line_dash="dash", line_color="red")
    figs['residuals'] = fig_residuals
    
    return figs


def run_forecast_demo():
    """Run the full supply forecasting demo."""
    from data.synthetic_data import generate_supply_forecast_data
    
    # Generate data
    df = generate_supply_forecast_data(n_days=365)
    
    # Train model
    model = SupplyForecastModel()
    metrics, test_results = model.train(df)
    
    print("Supply Forecasting Model Results")
    print("=" * 40)
    print(f"Train size: {metrics['train_size']} days")
    print(f"Test size: {metrics['test_size']} days")
    print(f"MAE: {metrics['mae']:,.0f} impressions")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"95% CI Coverage: {metrics['coverage']:.1f}%")
    
    return model, df, metrics


if __name__ == "__main__":
    run_forecast_demo()
