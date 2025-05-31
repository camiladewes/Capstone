from utils import load_datasets
from feature_pipeline import *
from modelling import train_lightgbm
from api_predictor import generate_features_for_api
import joblib
import pickle

# 1. Load datasets
product_prices, chain_campaigns, product_structures = load_datasets(
    "product_prices.csv", "chain_campaigns.csv", "product_structures.csv"
)

# 2. Create training features for each competitor
df_A = create_features("competitorA", product_prices, chain_campaigns, product_structures)
df_B = create_features("competitorB", product_prices, chain_campaigns, product_structures)

# 3. Split features and target variable
X_A = df_A.drop(columns=['pvp_was', 'time_key', 'sku'])
y_A = df_A['pvp_was']
X_B = df_B.drop(columns=['pvp_was', 'time_key', 'sku'])
y_B = df_B['pvp_was']

# 4. Save original dtypes for future inference
original_dtypes_A = X_A.dtypes.to_dict()
original_dtypes_B = X_B.dtypes.to_dict()
with open("original_dtypes_A.pkl", "wb") as f:
    pickle.dump(original_dtypes_A, f)
with open("original_dtypes_B.pkl", "wb") as f:
    pickle.dump(original_dtypes_B, f)

# 5. Split training and validation sets
from sklearn.model_selection import train_test_split
X_train_A, X_val_A, y_train_A, y_val_A = train_test_split(X_A, y_A, test_size=0.2, shuffle=False)
X_train_B, X_val_B, y_train_B, y_val_B = train_test_split(X_B, y_B, test_size=0.2, shuffle=False)

# 6. Train LightGBM models
modelA = train_lightgbm(X_train_A, y_train_A, X_val_A, y_val_A)
modelB = train_lightgbm(X_train_B, y_train_B, X_val_B, y_val_B)

# 7. Save models and structures
joblib.dump(modelA, "modelA.pkl")
joblib.dump(modelB, "modelB.pkl")

print("Models and structures saved successfully.")
