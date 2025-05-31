
chain_campaigns_path = 'Capstone/data/chain_campaigns.csv'
product_prices_path = 'Capstone/data/product_prices_leaflets.csv'
product_structures_path = 'Capstone/data/product_structures_sales.csv'

def load_datasets(product_prices_path, chain_campaigns_path, product_structures_path):
    import pandas as pd
    df_prices = pd.read_csv(product_prices_path)
    df_prices['time_key'] = pd.to_datetime(df_prices['time_key'], format='%Y%m%d')
    df_prices['sku'] = df_prices['sku'].astype(str)
    df_campaigns = pd.read_csv(chain_campaigns_path)
    df_campaigns['start_date'] = pd.to_datetime(df_campaigns['start_date'])
    df_campaigns['end_date'] = pd.to_datetime(df_campaigns['end_date'])
    df_struct = pd.read_csv(product_structures_path)
    df_struct['sku'] = df_struct['sku'].astype(str)
    return df_prices, df_campaigns, df_struct