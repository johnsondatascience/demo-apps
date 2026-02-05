"""
Campaign Performance Prediction Model
Predicts ROAS (Return on Ad Spend) for CTV advertising campaigns.
Demonstrates XGBoost modeling with feature importance analysis.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CampaignPerformanceModel:
    """
    XGBoost model for predicting campaign ROAS.
    Mirrors real-world ad tech campaign optimization workflows.
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_cols = ['vertical', 'device_type', 'daypart']
        self.numeric_cols = ['budget', 'duration_days', 'cpm', 'vcr', 'viewability']
        
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Preprocess features for model training/prediction."""
        df = df.copy()
        
        # Encode categorical variables
        for col in self.categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Select and scale features
        feature_cols = self.categorical_cols + self.numeric_cols
        X = df[feature_cols].values
        
        if fit:
            X = self.scaler.fit_transform(X)
            self.feature_names = feature_cols
        else:
            X = self.scaler.transform(X)
            
        return X
    
    def train(self, df: pd.DataFrame) -> dict:
        """Train the XGBoost model and return performance metrics."""
        X = self.preprocess(df, fit=True)
        y = df['roas'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions and metrics
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_scores': cross_val_score(self.model, X, y, cv=5, scoring='r2')
        }
        
        return metrics, X_test, y_test, y_pred
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ROAS for new campaigns."""
        X = self.preprocess(df, fit=False)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


def create_performance_visualizations(df: pd.DataFrame, metrics: dict, 
                                     y_test: np.ndarray, y_pred: np.ndarray,
                                     model: CampaignPerformanceModel) -> dict:
    """Create interactive visualizations for the campaign performance model."""
    
    figs = {}
    
    # 1. Feature Importance
    importance_df = model.get_feature_importance()
    fig_importance = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance for ROAS Prediction',
        labels={'importance': 'Importance Score', 'feature': 'Feature'}
    )
    fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
    figs['feature_importance'] = fig_importance
    
    # 2. Actual vs Predicted
    fig_scatter = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual ROAS', 'y': 'Predicted ROAS'},
        title=f'Actual vs Predicted ROAS (R² = {metrics["r2"]:.3f})'
    )
    fig_scatter.add_trace(
        go.Scatter(x=[0, max(y_test)], y=[0, max(y_test)], 
                  mode='lines', name='Perfect Prediction',
                  line=dict(dash='dash', color='red'))
    )
    figs['actual_vs_predicted'] = fig_scatter
    
    # 3. ROAS by Device Type
    fig_device = px.box(
        df, x='device_type', y='roas',
        title='ROAS Distribution by Device Type',
        color='device_type'
    )
    figs['roas_by_device'] = fig_device
    
    # 4. ROAS by Vertical
    vertical_roas = df.groupby('vertical')['roas'].agg(['mean', 'std']).reset_index()
    fig_vertical = px.bar(
        vertical_roas, x='vertical', y='mean',
        error_y='std',
        title='Average ROAS by Vertical',
        labels={'mean': 'Average ROAS', 'vertical': 'Vertical'}
    )
    figs['roas_by_vertical'] = fig_vertical
    
    # 5. Budget vs ROAS with device coloring
    fig_budget = px.scatter(
        df, x='budget', y='roas', color='device_type',
        title='Budget vs ROAS by Device Type',
        labels={'budget': 'Campaign Budget ($)', 'roas': 'ROAS'}
    )
    figs['budget_vs_roas'] = fig_budget
    
    return figs


def run_campaign_demo():
    """Run the full campaign performance demo."""
    from data.synthetic_data import generate_campaign_data
    
    # Generate data
    df = generate_campaign_data(n_campaigns=500)
    
    # Train model
    model = CampaignPerformanceModel()
    metrics, X_test, y_test, y_pred = model.train(df)
    
    print("Campaign Performance Model Results")
    print("=" * 40)
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"R²: {metrics['r2']:.3f}")
    print(f"CV R² (mean ± std): {metrics['cv_scores'].mean():.3f} ± {metrics['cv_scores'].std():.3f}")
    
    print("\nFeature Importance:")
    print(model.get_feature_importance())
    
    return model, df, metrics, y_test, y_pred


if __name__ == "__main__":
    run_campaign_demo()
