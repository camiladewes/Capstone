import pandas as pd
import numpy as np
import holidays

def generate_features_for_api(sku, target_date, product_prices, chain_campaigns,
                              product_structures, competitor, original_dtypes):
    """
    Generates features for a specific SKU and target date to be used in API predictions.
    Args:
        sku (str): SKU of the product.
        target_date (str or datetime): Target date for the prediction in 'YYYY-MM-DD' format.
        product_prices (pd.DataFrame): DataFrame containing product prices.
        chain_campaigns (pd.DataFrame): DataFrame containing chain campaigns.
        product_structures (pd.DataFrame): DataFrame containing product structures.
        competitor (str): Competitor name to filter prices.
        original_dtypes (dict): Original dtypes of the training data for alignment.
    """
    sku = str(sku)
    target_date = pd.to_datetime(target_date)

    # Padronizar tipos para evitar conflitos
    product_prices = product_prices.copy()
    product_prices['sku'] = product_prices['sku'].astype(str)
    product_structures = product_structures.copy()
    product_structures['sku'] = product_structures['sku'].astype(str)

    last_obs_date = product_prices['time_key'].max()

    history = product_prices[
        (product_prices['sku'] == sku) &
        (product_prices['time_key'] <= last_obs_date) &
        (product_prices['competitor'] == competitor)
    ].copy()

    row = {
        'sku': sku,
        'time_key': target_date,
        'competitor': competitor,
        'leaflet': None,
        'pvp_was': np.nan
    }
    df_target = pd.DataFrame([row])

    df_all = pd.concat([history, df_target], ignore_index=True)
    df_all['time_key'] = pd.to_datetime(df_all['time_key'])
    df_all = df_all.sort_values(['sku', 'time_key'])

    df_all = add_temporal_features(df_all)
    df_all = add_product_category(df_all, product_structures)
    df_all = add_campaign_features(df_all, chain_campaigns, competitor)

    df_all['leaflet'] = df_all['leaflet'].fillna('unknown')
    df_all = encode_leaflet(df_all)
    df_all = add_time_series_features(df_all)
    df_all = add_competitor_prices(df_all, product_prices, ['chain', 'competitorA', 'competitorB'])
    df_all = additional_features(df_all, product_prices, current_competitor=competitor)

    df_pred = df_all[df_all['time_key'] == target_date].copy()
    numeric_cols = df_pred.select_dtypes(include=[np.number]).columns
    df_pred[numeric_cols] = df_pred[numeric_cols].fillna(0)

    # Align with training data 
    cols_to_keep = original_dtypes.keys()
    df_pred = df_pred[[col for col in df_pred.columns if col in cols_to_keep]]
    df_pred = df_pred.reindex(columns=cols_to_keep)

    for col, dtype in original_dtypes.items():
        if col in df_pred.columns:
            if isinstance(dtype, pd.CategoricalDtype):
                df_pred[col] = df_pred[col].astype('category')
            else:
                df_pred[col] = df_pred[col].astype(dtype)

    final_drop_cols = ['pvp_was', 'time_key', 'sku', 'competitor']
    df_pred = df_pred.drop(columns=[col for col in final_drop_cols if col in df_pred.columns], errors='ignore')

    return df_pred
