from flask import Flask, request, jsonify
from peewee import SqliteDatabase, Model, CharField, IntegerField, DoubleField
from playhouse.shortcuts import model_to_dict
import lightgbm as lgb
import pandas as pd
import pickle
import holidays
from datetime import datetime, timedelta
import numpy as np

# Configuração inicial
app = Flask(__name__)
db = SqliteDatabase('price_predictions.db')
pt_holidays = holidays.Portugal()

# Carregar modelos e features
modelA = lgb.Booster(model_file='modelA.txt')
modelB = lgb.Booster(model_file='modelB.txt')

with open('features_A.pkl', 'rb') as f:
    features_A = pickle.load(f)
    
with open('features_B.pkl', 'rb') as f:
    features_B = pickle.load(f)

# Modelo de banco de dados para previsões
class PricePrediction(Model):
    sku = CharField()
    time_key = IntegerField()
    pvp_is_competitorA = DoubleField(null=True)
    pvp_is_competitorB = DoubleField(null=True)
    pvp_is_competitorA_actual = DoubleField(null=True)
    pvp_is_competitorB_actual = DoubleField(null=True)
    created_at = IntegerField(default=lambda: int(datetime.now().timestamp()))
    
    class Meta:
        database = db
        indexes = (
            (('sku', 'time_key'), True),
        )

db.create_tables([PricePrediction], safe=True)

# Funções auxiliares para feature engineering
def get_historical_data(sku, time_key, days_back=30):
    """Busca dados históricos do SKU"""
    # Implemente a conexão com seu banco de dados histórico aqui
    # Retornar DataFrame com colunas: time_key, pvp_was, competitor_prices, etc.
    # Exemplo simplificado:
    return pd.DataFrame({
        'time_key': [time_key - timedelta(days=i) for i in range(days_back, 0, -1)],
        'pvp_was': np.random.rand(days_back) * 100  # Dummy data
    })

def calculate_features(sku, time_key, competitor):
    """Calcula todas as features necessárias"""
    # Converter time_key para datetime
    current_date = datetime.strptime(str(time_key), '%Y%m%d')
    
    # Buscar dados históricos
    hist_data = get_historical_data(sku, current_date)
    
    # Calcular features temporais
    features = {
        'sku': sku,
        'time_key': time_key,
        'day_of_month': current_date.day,
        'day_of_week': current_date.weekday(),
        'month': current_date.month,
        'holiday_flag': int(current_date in pt_holidays)
    }
    
    # Calcular lags e médias móveis
    for lag in [7, 14, 30]:
        features[f'lag_{lag}'] = hist_data['pvp_was'].iloc[-lag] if len(hist_data) >= lag else np.nan
    
    for window in [1, 7, 14, 30]:
        features[f'rolling_mean_{window}'] = hist_data['pvp_was'].tail(window).mean()
    
    # Adicionar outras features conforme seu pipeline
    # ...
    
    return pd.DataFrame([features], columns=features_A if competitor == 'A' else features_B)

@app.route('/forecast_prices/', methods=['POST'])
def forecast_prices():
    try:
        data = request.get_json()
        
        # Validação
        if 'sku' not in data or 'time_key' not in data:
            return jsonify({'error': 'Missing required fields'}), 422
            
        sku = str(data['sku'])
        time_key = int(data['time_key'])
        
        # Verificar se já existe previsão
        try:
            existing = PricePrediction.get(
                (PricePrediction.sku == sku) & 
                (PricePrediction.time_key == time_key))
            return jsonify({
                'sku': sku,
                'time_key': time_key,
                'pvp_is_competitorA': existing.pvp_is_competitorA,
                'pvp_is_competitorB': existing.pvp_is_competitorB
            })
        except PricePrediction.DoesNotExist:
            # Gerar features
            df_A = calculate_features(sku, time_key, 'A')
            df_B = calculate_features(sku, time_key, 'B')
            
            # Fazer previsões
            pred_A = float(modelA.predict(df_A)[0])
            pred_B = float(modelB.predict(df_B)[0])
            
            # Armazenar previsão
            PricePrediction.create(
                sku=sku,
                time_key=time_key,
                pvp_is_competitorA=pred_A,
                pvp_is_competitorB=pred_B
            )
            
            return jsonify({
                'sku': sku,
                'time_key': time_key,
                'pvp_is_competitorA': pred_A,
                'pvp_is_competitorB': pred_B
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/actual_prices/', methods=['POST'])
def actual_prices():
    try:
        data = request.get_json()
        
        # Validação
        required_fields = ['sku', 'time_key', 'pvp_is_competitorA_actual', 'pvp_is_competitorB_actual']
        if not all(f in data for f in required_fields):
            return jsonify({'error': 'Missing required fields'}), 422
            
        # Atualizar registro
        record = PricePrediction.get(
            (PricePrediction.sku == data['sku']) & 
            (PricePrediction.time_key == data['time_key']))
        
        record.pvp_is_competitorA_actual = float(data['pvp_is_competitorA_actual'])
        record.pvp_is_competitorB_actual = float(data['pvp_is_competitorB_actual'])
        record.save()
        
        return jsonify(model_to_dict(record))
        
    except PricePrediction.DoesNotExist:
        return jsonify({'error': 'Prediction not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)