
def create_features_dask(competitor, product_prices, chain_campaigns, product_structures, 
                        other_competitors=['competitorB', 'chain'], npartitions=None):
    """
    Versão otimizada com Dask para processamento de grandes datasets
    
    Args:
        competitor: Nome do competidor a ser analisado
        product_prices: DataFrame com preços dos produtos
        chain_campaigns: DataFrame com campanhas
        product_structures: DataFrame com estruturas de produtos
        other_competitors: Lista de outros competidores
        npartitions: Número de partições para o Dask (None para automático)
    """
    # Converter para Dask DataFrame
    ddf = dd.from_pandas(product_prices, npartitions=npartitions)
    
    # Filtrar pelo competidor
    ddf = ddf[ddf['competitor'] == competitor].copy()
    
    # Converter tipos básicos
    ddf['sku'] = ddf['sku'].astype(str)
    ddf['time_key'] = dd.to_datetime(ddf['time_key'])
    
    # Ordenar - importante para operações temporais
    ddf = ddf.set_index('time_key').reset_index()  # Garantir ordenação
    
    # Adicionar features temporais
    def _add_temporal_features(df):
        pt_holidays = holidays.Portugal()
        df['day_of_month'] = df['time_key'].dt.day
        df['day_of_week'] = df['time_key'].dt.dayofweek
        df['month'] = df['time_key'].dt.month
        df['holiday_flag'] = df['time_key'].isin(pt_holidays).astype(int)
        return df
    
    ddf = ddf.map_partitions(_add_temporal_features)
    
    # Codificar leaflet (categoria)
    def _encode_leaflet(df):
        leaflet_mapping = {'themed': 1, 'weekly': 2, 'short': 3, 'unknown': 0}
        df['leaflet'] = df['leaflet'].map(leaflet_mapping).fillna(0).astype('int8')
        return df
    
    ddf = ddf.map_partitions(_encode_leaflet)
    
    # Adicionar features de campanha (precisa de join com pandas)
    def _add_campaign_features(df, chain_campaigns, competitor):
        # Processamento pequeno pode ser feito em pandas
        campaigns = chain_campaigns[chain_campaigns['competitor'] == competitor][['start_date', 'end_date', 'chain_campaign']].drop_duplicates()
        date_ranges = []
        for _, row in campaigns.iterrows():
            dates = pd.date_range(row['start_date'], row['end_date'], freq='D')
            date_ranges.extend([(date, row['chain_campaign']) for date in dates])
        campaign_dates = pd.DataFrame(date_ranges, columns=['time_key', 'campaign_code'])
        
        # Converter para dask para o merge
        campaign_dates_dask = dd.from_pandas(campaign_dates, npartitions=1)
        df = df.merge(campaign_dates_dask, on='time_key', how='left')
        df['campaign_active'] = df['campaign_code'].notna().astype('int8')
        df['campaign_type'] = df['campaign_code'].str.extract(r'([A-Za-z]+)')[0].astype('category').cat.codes.add(1).fillna(0).astype('int8')
        return df.drop(columns=['campaign_code'])
    
    ddf = ddf.map_partitions(_add_campaign_features, chain_campaigns, competitor)
    
    # Adicionar categoria de produto
    product_structures_dask = dd.from_pandas(product_structures.astype({'sku': str}), npartitions=1)
    ddf = ddf.merge(product_structures_dask[['sku', 'structure_level_2']], on='sku', how='left')
    
    # Features de séries temporais (rolling) - desafio no Dask
    def _add_time_series_features(df):
        for w in [7, 14, 30]:
            df[f'rolling_std_{w}'] = df.groupby('sku')['pvp_was'].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
        return df
    
    # Aplicar por partição (menos eficiente, mas funciona)
    ddf = ddf.map_partitions(_add_time_series_features)
    
    # Preços dos concorrentes
    for comp in other_competitors:
        comp_prices = product_prices[product_prices['competitor'] == comp]
        comp_prices_dask = dd.from_pandas(
            comp_prices.rename(columns={'pvp_was': f'pvp_was_{comp}'})[['time_key', 'sku', f'pvp_was_{comp}']],
            npartitions=npartitions
        )
        ddf = ddf.merge(comp_prices_dask, on=['time_key', 'sku'], how='left')
    
    # Features adicionais
    def _additional_features(df, current_competitor):
        for competitor in other_competitors:
            col_name = f'pvp_was_{competitor}'
            delta_col = f'delta_price_{competitor}'
            if col_name in df.columns:
                df[delta_col] = df['pvp_was'] - df[col_name]
        
        comparison_cols = [f'pvp_was_{c}' for c in other_competitors if f'pvp_was_{c}' in df.columns]
        if len(comparison_cols) > 0:
            df['is_cheapest'] = (df['pvp_was'] < df[comparison_cols].min(axis=1)).astype('int8')
            df['is_most_expensive'] = (df['pvp_was'] > df[comparison_cols].max(axis=1)).astype('int8')
            for competitor in other_competitors:
                col = f'pvp_was_{competitor}'
                if col in df.columns:
                    df[f'is_cheaper_than_{competitor}'] = (df['pvp_was'] < df[col]).astype('int8')
        return df
    
    ddf = ddf.map_partitions(_additional_features, competitor)
    
    # Remover NAs e resetar índice
    ddf = ddf.dropna(subset=['pvp_was']).fillna(0)
    
    # Computar o resultado final (opcional - pode retornar o dask dataframe)
    with ProgressBar():
        result = ddf.compute()
    
    return result.reset_index(drop=True)