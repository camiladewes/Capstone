# api.py
import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
import pickle
import pandas as pd
from datetime import datetime
from api_predictor import generate_features_for_api

# Carregar variáveis de ambiente
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///forecast_prices.db')

# Inicializa app Flask e banco de dados (Postgres ou SQLite)
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define modelo da tabela no banco
class Forecast(db.Model):
    __tablename__ = 'forecast_prices'
    sku = db.Column(db.String, primary_key=True)
    time_key = db.Column(db.Integer, primary_key=True)
    pvp_is_competitorA = db.Column(db.Float)
    pvp_is_competitorB = db.Column(db.Float)
    pvp_is_competitorA_actual = db.Column(db.Float, nullable=True)
    pvp_is_competitorB_actual = db.Column(db.Float, nullable=True)

# Carrega modelos e dtypes apenas uma vez (fora do handler)
modelA = joblib.load("modelA.pkl")
modelB = joblib.load("modelB.pkl")
with open("original_dtypes_A.pkl", "rb") as f:
    original_dtypes_A = pickle.load(f)
with open("original_dtypes_B.pkl", "rb") as f:
    original_dtypes_B = pickle.load(f)

# Carrega dados históricos uma vez
product_prices = pd.read_csv("data/product_prices_leaflets.csv")
product_prices['time_key'] = pd.to_datetime(product_prices['time_key'])
product_prices['sku'] = product_prices['sku'].astype(str)
chain_campaigns = pd.read_csv("data/chain_campaigns.csv")
chain_campaigns['start_date'] = pd.to_datetime(chain_campaigns['start_date'])
chain_campaigns['end_date'] = pd.to_datetime(chain_campaigns['end_date'])
product_structures = pd.read_csv("data/product_structures_sales.csv")
product_structures['sku'] = product_structures['sku'].astype(str)

@app.route("/forecast_prices/", methods=["POST"])
def forecast_prices():
    data = request.get_json()
    try:
        # Validação básica de input
        if 'sku' not in data or 'time_key' not in data:
            raise ValueError("Campos 'sku' e 'time_key' são obrigatórios")
        sku = str(data["sku"])
        time_key = int(data["time_key"])
        # Converter time_key de inteiro (timestamp) para datetime
        target_date = pd.to_datetime(time_key, origin='unix', unit='D')

        # Gera features previamente carregados em memória
        X_A = generate_features_for_api(sku, target_date, product_prices, chain_campaigns,
                                        product_structures, 'competitorA', original_dtypes_A)
        X_B = generate_features_for_api(sku, target_date, product_prices, chain_campaigns,
                                        product_structures, 'competitorB', original_dtypes_B)
        if X_A is None or X_B is None or X_A.empty or X_B.empty:
            raise ValueError("Não há dados históricos suficientes para gerar a predição")

        # Predição
        y_pred_A = modelA.predict(X_A)[0]
        y_pred_B = modelB.predict(X_B)[0]

        # Armazenar no banco, garantindo unicidade (sku, time_key)
        existing = Forecast.query.filter_by(sku=sku, time_key=time_key).first()
        if not existing:
            new_forecast = Forecast(
                sku=sku,
                time_key=time_key,
                pvp_is_competitorA=y_pred_A,
                pvp_is_competitorB=y_pred_B
            )
            db.session.add(new_forecast)
            db.session.commit()

        return jsonify({
            "sku": sku,
            "time_key": time_key,
            "pvp_is_competitorA": float(y_pred_A),
            "pvp_is_competitorB": float(y_pred_B)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 422

@app.route("/actual_prices/", methods=["POST"])
def actual_prices():
    data = request.get_json()
    try:
        if 'sku' not in data or 'time_key' not in data or 'pvp_is_competitorA_actual' not in data or 'pvp_is_competitorB_actual' not in data:
            raise ValueError("Campos obrigatórios: sku, time_key, pvp_is_competitorA_actual, pvp_is_competitorB_actual")
        sku = str(data["sku"])
        time_key = int(data["time_key"])
        actual_A = float(data["pvp_is_competitorA_actual"])
        actual_B = float(data["pvp_is_competitorB_actual"])

        record = Forecast.query.filter_by(sku=sku, time_key=time_key).first()
        if not record:
            raise ValueError("Produto e data não encontrados no banco")

        record.pvp_is_competitorA_actual = actual_A
        record.pvp_is_competitorB_actual = actual_B
        db.session.commit()

        return jsonify({
            "sku": sku,
            "time_key": time_key,
            "pvp_is_competitorA": record.pvp_is_competitorA,
            "pvp_is_competitorB": record.pvp_is_competitorB,
            "pvp_is_competitorA_actual": actual_A,
            "pvp_is_competitorB_actual": actual_B
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 422

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
