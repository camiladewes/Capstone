# feature_pipeline.py
import pandas as pd
import numpy as np
import holidays

def add_temporal_features(df):
    pt_holidays = holidays.Portugal()
    df['day_of_month'] = df['time_key'].dt.day
    df['day_of_week'] = df['time_key'].dt.dayofweek
    df['month'] = df['time_key'].dt.month
    df['holiday_flag'] = df['time_key'].isin(pt_holidays).astype(int)
    return df

def encode_leaflet(df):
    leaflet_mapping = {'themed': 1, 'weekly': 2, 'short': 3, 'unknown': 0}
    df['leaflet'] = df['leaflet'].map(leaflet_mapping).fillna(0).astype('category')
    return df

def add_campaign_features(df, chain_campaigns, competitor):
    campaigns = chain_campaigns[chain_campaigns['competitor'] == competitor][['start_date', 'end_date', 'chain_campaign']].drop_duplicates()
    date_ranges = []
    for _, row in campaigns.iterrows():
        dates = pd.date_range(row['start_date'], row['end_date'], freq='D')
        date_ranges.extend([(date, row['chain_campaign']) for date in dates])
    campaign_dates = pd.DataFrame(date_ranges, columns=['time_key', 'campaign_code'])
    df = df.merge(campaign_dates, on='time_key', how='left')
    df['campaign_active'] = df['campaign_code'].notna().astype(int)
    df['campaign_type'] = df['campaign_code'].str.extract(r'([A-Za-z]+)')[0].astype('category').cat.codes.add(1).fillna(0).astype(int)
    return df.drop(columns=['campaign_code'])

def add_product_category(df, product_structures):
    product_structures = product_structures.copy()
    product_structures['sku'] = product_structures['sku'].astype(str)
    return df.merge(product_structures[['sku', 'structure_level_2']], on='sku', how='left')

def add_time_series_features(df):
    grouper = df.groupby('sku')
    for lag in [7, 14, 30]:
        df[f'lag_{lag}'] = grouper['pvp_was'].shift(lag)
    for w in [1, 7, 14, 30]:
        df[f'rolling_mean_{w}'] = grouper['pvp_was'].transform(lambda x: x.rolling(w, min_periods=1).mean())
    for w in [7, 14, 30]:
        df[f'rolling_std_{w}'] = grouper['pvp_was'].transform(lambda x: x.rolling(w, min_periods=1).std())
    return df

def add_competitor_prices(df, product_prices, other_competitors):
    product_prices = product_prices.copy()
    product_prices['sku'] = product_prices['sku'].astype(str)
    for comp in other_competitors:
        comp_prices = product_prices[product_prices['competitor'] == comp].rename(columns={'pvp_was': f'pvp_was_{comp}'})[['time_key', 'sku', f'pvp_was_{comp}']]
        df = df.merge(comp_prices, on=['time_key', 'sku'], how='left')
    return df

def additional_features(df, product_prices, current_competitor):
    product_prices = product_prices.copy()
    product_prices['sku'] = product_prices['sku'].astype(str)
    all_competitors = product_prices['competitor'].unique()
    other_competitors = [c for c in all_competitors if c != current_competitor and c in ['chain', 'competitorA', 'competitorB']]
    for competitor in other_competitors:
        col_name = f'pvp_was_{competitor}'
        delta_col = f'delta_price_{competitor}'
        if col_name in df.columns:
            df[delta_col] = df['pvp_was'] - df[col_name]
    for competitor in other_competitors:
        comp_data = product_prices[product_prices['competitor'] == competitor].copy()
        comp_data['time_key'] = pd.to_datetime(comp_data['time_key'])
        comp_data = comp_data.sort_values(['sku', 'time_key'])
        comp_data[f'lag1_price_{competitor}'] = comp_data.groupby('sku')['pvp_was'].transform(lambda x: x.shift(1))
        comp_data[f'lag7_price_{competitor}'] = comp_data.groupby('sku')['pvp_was'].transform(lambda x: x.shift(7))
        lag_cols = ['time_key', 'sku', f'lag1_price_{competitor}', f'lag7_price_{competitor}']
        df = df.merge(comp_data[lag_cols], on=['time_key', 'sku'], how='left')
        if f'lag1_price_{competitor}' in df.columns:
            df[f'delta_{competitor}_lag1'] = df['pvp_was'] - df[f'lag1_price_{competitor}']
        if f'lag7_price_{competitor}' in df.columns:
            df[f'delta_{competitor}_lag7'] = df['pvp_was'] - df[f'lag7_price_{competitor}']
    comparison_cols = [f'pvp_was_{c}' for c in other_competitors if f'pvp_was_{c}' in df.columns]
    if len(comparison_cols) > 0:
        df['is_cheapest'] = (df['pvp_was'] < df[comparison_cols].min(axis=1)).astype(int)
        df['is_most_expensive'] = (df['pvp_was'] > df[comparison_cols].max(axis=1)).astype(int)
        for competitor in other_competitors:
            col = f'pvp_was_{competitor}'
            if col in df.columns:
                df[f'is_cheaper_than_{competitor}'] = (df['pvp_was'] < df[col]).astype(int)
        price_cols = ['pvp_was'] + comparison_cols
        df['price_rank'] = df[price_cols].rank(axis=1, method='min', na_option='bottom')['pvp_was']
    return df

def create_features(competitor, product_prices, chain_campaigns, product_structures, other_competitors=['competitorB', 'chain']):
    df = product_prices[product_prices['competitor'] == competitor].copy()
    df['sku'] = df['sku'].astype(str)
    df['time_key'] = pd.to_datetime(df['time_key'])
    df = df.sort_values(['sku', 'time_key'])
    df = add_temporal_features(df)
    df = add_time_series_features(df)
    df = encode_leaflet(df)
    df = add_campaign_features(df, chain_campaigns, competitor)
    df = add_product_category(df, product_structures)
    df = add_competitor_prices(df, product_prices, other_competitors)
    df = additional_features(df, product_prices, current_competitor=competitor)
    df = df.dropna(subset=['pvp_was']).fillna(0).reset_index(drop=True)
    return df
