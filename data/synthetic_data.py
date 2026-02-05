"""
Synthetic Ad Tech Data Generator
Generates realistic CTV/programmatic advertising datasets for demo purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_campaign_data(n_campaigns: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic campaign performance data.
    Simulates CTV programmatic advertising campaigns with realistic metrics.
    """
    np.random.seed(seed)
    
    # Campaign attributes
    verticals = ['Automotive', 'Retail', 'Finance', 'CPG', 'Travel', 'Healthcare', 'Tech', 'Entertainment']
    device_types = ['CTV', 'Mobile', 'Desktop', 'Tablet']
    dayparts = ['Morning', 'Afternoon', 'Primetime', 'Late Night']
    
    campaigns = []
    for i in range(n_campaigns):
        vertical = np.random.choice(verticals)
        device = np.random.choice(device_types, p=[0.45, 0.30, 0.15, 0.10])  # CTV-heavy
        daypart = np.random.choice(dayparts)
        
        # Base metrics with realistic correlations
        budget = np.random.lognormal(mean=10, sigma=1)  # $2K - $200K range
        duration_days = np.random.randint(7, 90)
        
        # CTV tends to have higher CPMs but better engagement
        base_cpm = {'CTV': 25, 'Mobile': 12, 'Desktop': 8, 'Tablet': 10}[device]
        cpm = base_cpm * np.random.uniform(0.7, 1.5)
        
        impressions = (budget / cpm) * 1000
        
        # Click-through rate varies by device and vertical
        base_ctr = {'CTV': 0.008, 'Mobile': 0.012, 'Desktop': 0.015, 'Tablet': 0.011}[device]
        vertical_mult = {'Automotive': 1.2, 'Retail': 1.1, 'Finance': 0.9, 'CPG': 1.0, 
                        'Travel': 1.3, 'Healthcare': 0.8, 'Tech': 1.1, 'Entertainment': 1.4}[vertical]
        ctr = base_ctr * vertical_mult * np.random.uniform(0.5, 1.5)
        
        clicks = int(impressions * ctr)
        
        # Conversion rate
        base_cvr = 0.025 * np.random.uniform(0.5, 2.0)
        conversions = int(clicks * base_cvr)
        
        # Video completion rate (CTV-specific, important metric)
        vcr = np.random.beta(8, 2) if device == 'CTV' else np.random.beta(5, 3)
        
        # Viewability
        viewability = np.random.beta(7, 3)
        
        # Cost efficiency metrics
        cpc = budget / max(clicks, 1)
        cpa = budget / max(conversions, 1)
        
        # ROAS (Return on Ad Spend) - key outcome variable
        revenue_per_conversion = np.random.lognormal(mean=4, sigma=0.8)
        revenue = conversions * revenue_per_conversion
        roas = revenue / budget
        
        campaigns.append({
            'campaign_id': f'CMP_{i:05d}',
            'vertical': vertical,
            'device_type': device,
            'daypart': daypart,
            'budget': budget,
            'duration_days': duration_days,
            'impressions': int(impressions),
            'clicks': clicks,
            'conversions': conversions,
            'cpm': cpm,
            'ctr': ctr,
            'cvr': base_cvr,
            'vcr': vcr,
            'viewability': viewability,
            'cpc': cpc,
            'cpa': cpa,
            'revenue': revenue,
            'roas': roas
        })
    
    return pd.DataFrame(campaigns)


def generate_ab_test_data(n_users: int = 10000, true_lift: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic A/B test data for incrementality measurement.
    Simulates a CTV ad exposure experiment with conversion outcomes.
    """
    np.random.seed(seed)
    
    # User characteristics
    users = []
    for i in range(n_users):
        # Treatment assignment (50/50 split)
        treatment = np.random.binomial(1, 0.5)
        
        # User propensity (baseline conversion probability)
        propensity = np.random.beta(2, 20)  # ~10% baseline
        
        # Treatment effect (lift for exposed users)
        if treatment:
            conversion_prob = propensity * (1 + true_lift)
        else:
            conversion_prob = propensity
        
        converted = np.random.binomial(1, min(conversion_prob, 1.0))
        
        # Ad exposures (for exposed group)
        exposures = np.random.poisson(5) if treatment else 0
        
        # Time to conversion (if converted)
        days_to_convert = np.random.exponential(7) if converted else np.nan
        
        users.append({
            'user_id': f'USR_{i:07d}',
            'treatment': treatment,
            'exposures': exposures,
            'converted': converted,
            'baseline_propensity': propensity,
            'days_to_convert': days_to_convert
        })
    
    return pd.DataFrame(users)


def generate_supply_forecast_data(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic ad inventory (supply) time series data.
    Simulates daily available impressions with seasonality and trends.
    """
    np.random.seed(seed)
    
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    data = []
    for i, date in enumerate(dates):
        # Base supply (100M impressions/day)
        base_supply = 100_000_000
        
        # Trend (5% growth over year)
        trend = 1 + (i / n_days) * 0.05
        
        # Weekly seasonality (weekends have 20% more CTV viewing)
        day_of_week = date.weekday()
        weekly_factor = 1.2 if day_of_week >= 5 else 1.0
        
        # Monthly seasonality (Q4 higher due to holidays)
        month = date.month
        monthly_factor = {1: 0.9, 2: 0.85, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.0,
                         7: 0.95, 8: 0.95, 9: 1.0, 10: 1.05, 11: 1.15, 12: 1.25}[month]
        
        # Holiday effects
        is_holiday = (month == 12 and date.day >= 20) or (month == 11 and date.day >= 22 and date.day <= 28)
        holiday_factor = 1.3 if is_holiday else 1.0
        
        # Noise
        noise = np.random.normal(1, 0.05)
        
        # Calculate supply
        supply = base_supply * trend * weekly_factor * monthly_factor * holiday_factor * noise
        
        # Fill rate (how much inventory is sold)
        base_fill = 0.75
        fill_rate = min(base_fill * monthly_factor * np.random.uniform(0.9, 1.1), 0.95)
        
        # CPM (price increases when demand > supply)
        base_cpm = 22
        demand_pressure = fill_rate / base_fill
        cpm = base_cpm * demand_pressure * np.random.uniform(0.9, 1.1)
        
        data.append({
            'date': date,
            'available_impressions': int(supply),
            'filled_impressions': int(supply * fill_rate),
            'fill_rate': fill_rate,
            'avg_cpm': cpm,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': day_of_week >= 5,
            'is_holiday_period': is_holiday
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate and save sample data
    campaigns = generate_campaign_data()
    print(f"Campaign data: {len(campaigns)} records")
    print(campaigns.head())
    
    ab_test = generate_ab_test_data()
    print(f"\nA/B test data: {len(ab_test)} records")
    print(ab_test.head())
    
    supply = generate_supply_forecast_data()
    print(f"\nSupply data: {len(supply)} records")
    print(supply.head())
