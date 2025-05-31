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

def add_campaign_features(df, chain_campaigns):
    campaign_type_mapping = {
        'C1': 1,
        'C2': 2,
        'A1': 3,
        'A2': 4,
        'A3': 5
    }
    campaign_dates_df = chain_campaigns.groupby(['competitor', 'start_date', 'end_date', 'chain_campaign']) \
        .first().reset_index()
    all_campaigns = []
    for _, row in campaign_dates_df.iterrows():
        dates = pd.date_range(row['start_date'], row['end_date'], freq='D')
        for date in dates:
            all_campaigns.append({
                'time_key': date,
                'competitor': row['competitor'],
                'campaign_code': row['chain_campaign']
            })
    campaign_dates_df = pd.DataFrame(all_campaigns)
    campaign_dates_df = campaign_dates_df.drop_duplicates(
        subset=['time_key', 'competitor'], 
        keep='last' 
    )
    df = df.merge(campaign_dates_df, on=['time_key', 'competitor'], how='left')
    df['campaign_active'] = df['campaign_code'].notna().astype(int)
    df['campaign_type'] = df['campaign_code'].map(campaign_type_mapping).fillna(0).astype(int)
    return df.drop(columns=['campaign_code'], errors='ignore')

def add_product_category_optimized(df, product_structures):
    category_map = product_structures.drop_duplicates('sku').set_index('sku')['structure_level_2'].to_dict()
    df['structure_level_2'] = df['sku'].map(category_map)
    return df


def add_time_series_features(df):
    df = df.sort_values(['competitor', 'sku', 'time_key'])
    results = []
    # Processar cada grupo separadamente
    for (competitor, sku), group in df.groupby(['competitor', 'sku']):
        temp_df = group.copy()        
        for w in [7, 14, 30]:
            temp_df[f'rolling_std_{w}'] = temp_df['pvp_was'].rolling(w, min_periods=1).std()
            # 1. Preenche com média móvel expansiva do mesmo produto/competidor
            temp_df[f'rolling_std_{w}'] = temp_df[f'rolling_std_{w}'].fillna(
                temp_df['pvp_was'].expanding().std()
            )
            # 2. Preenche com std global do produto/competidor
            temp_df[f'rolling_std_{w}'] = temp_df[f'rolling_std_{w}'].fillna(
                temp_df['pvp_was'].std()
            )
            # 3. Último recurso: zero
            temp_df[f'rolling_std_{w}'] = temp_df[f'rolling_std_{w}'].fillna(0)
        results.append(temp_df)
    return pd.concat(results).sort_index()

def add_competitor_prices(df, product_prices, other_competitors):
    product_prices = product_prices.copy()
    product_prices['sku'] = product_prices['sku'].astype(str)
    for comp in other_competitors:
        comp_prices = product_prices[product_prices['competitor'] == comp].rename(columns={'pvp_was': f'pvp_was_{comp}'})[['time_key', 'sku', f'pvp_was_{comp}']]
        df = df.merge(comp_prices, on=['time_key', 'sku'], how='left')
    return df

def additional_features(df, product_prices, current_competitor):
    """
    Adiciona features competitivas SEM criar novas linhas, tratando:
    - Duplicatas nas chaves de merge
    - NaN values de forma apropriada
    - Mantendo a integridade do dataset original
    """
    # Criar cópia para não modificar o original
    df = df.copy()
    
    # Pré-processamento básico
    product_prices = product_prices.copy()
    product_prices['sku'] = product_prices['sku'].astype(str)
    df['sku'] = df['sku'].astype(str)
    
    # Identificar competidores relevantes
    valid_competitors = ['chain', 'competitorA', 'competitorB']
    other_competitors = [c for c in valid_competitors if c != current_competitor]
    
    # 1. Lidar com possíveis duplicatas no DataFrame principal
    if df.duplicated(subset=['time_key', 'sku']).any():
        print(f"Aviso: DataFrame principal tem {df.duplicated(subset=['time_key', 'sku']).sum()} linhas duplicadas - consolidando...")
        df = df.groupby(['time_key', 'sku']).first().reset_index()
    
    # 2. Cálculo de deltas de preço
    for competitor in other_competitors:
        col_name = f'pvp_was_{competitor}'
        
        if col_name in df.columns:
            # Preencher NaN com último valor conhecido para o mesmo SKU
            df[col_name] = df.groupby('sku')[col_name].ffill()
            
            # Calcular delta com tratamento seguro
            df[f'delta_price_{competitor}'] = df['pvp_was'] - df[col_name].fillna(0)
            df[f'{competitor}_price_missing'] = df[col_name].isna().astype(int)

    # 3. Features de lag (preços históricos)
    for competitor in other_competitors:
        # Processar dados do competidor - garantir sem duplicatas
        lag_data = (
            product_prices[product_prices['competitor'] == competitor]
            .drop_duplicates(subset=['time_key', 'sku'])
            .sort_values(['sku', 'time_key'])
        )
        
        # Calcular lags
        lag_data[f'lag1_price_{competitor}'] = lag_data.groupby('sku')['pvp_was'].shift(1)
        lag_data[f'lag7_price_{competitor}'] = lag_data.groupby('sku')['pvp_was'].shift(7)
        
        # Fazer merge seguro (mantendo número de linhas original)
        df = pd.merge(
            df,
            lag_data[['time_key', 'sku', f'lag1_price_{competitor}', f'lag7_price_{competitor}']],
            on=['time_key', 'sku'],
            how='left'
        )
        
        # Calcular deltas de lag
        for lag in [1, 7]:
            lag_col = f'lag{lag}_price_{competitor}'
            if lag_col in df.columns:
                df[f'delta_{competitor}_lag{lag}'] = df['pvp_was'] - df[lag_col].fillna(0)
                df[lag_col] = df[lag_col].fillna(0)

    # 4. Features de posicionamento competitivo
    comparison_cols = [f'pvp_was_{c}' for c in other_competitors if f'pvp_was_{c}' in df.columns]
    
    if comparison_cols:
        # Preencher NaN com valores altos para comparação
        price_matrix = df[comparison_cols].fillna(1e12)  # Número grande para ser considerado "mais caro"
        
        # Flags de posicionamento
        df['is_cheapest'] = (df['pvp_was'] < price_matrix.min(axis=1)).astype(int)
        df['is_most_expensive'] = (df['pvp_was'] > price_matrix.max(axis=1)).astype(int)
        
        for competitor in other_competitors:
            col = f'pvp_was_{competitor}'
            if col in df.columns:
                df[f'is_cheaper_than_{competitor}'] = (df['pvp_was'] < df[col].fillna(1e12)).astype(int)
        
        # Cálculo de ranking
        valid_prices = df[['pvp_was'] + comparison_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if not valid_prices.empty:
            ranks = valid_prices.rank(axis=1, method='min')
            df['price_rank'] = ranks['pvp_was'].reindex(df.index, fill_value=len(comparison_cols)+1)
        else:
            df['price_rank'] = len(comparison_cols) + 1
    
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
