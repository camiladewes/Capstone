from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pydantic import BaseModel, ValidationError, conint
from typing import Optional
import os
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Configuração do banco de dados
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelos Pydantic para validação
class ForecastInput(BaseModel):
    sku: str
    time_key: conint(gt=0)  # time_key deve ser um inteiro positivo

class ActualPriceInput(BaseModel):
    sku: str
    time_key: conint(gt=0)
    pvp_is_competitorA_actual: float
    pvp_is_competitorB_actual: float

# Modelo de banco de dados
class PricePrediction(db.Model):
    __tablename__ = 'price_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    sku = db.Column(db.String(100), nullable=False)
    time_key = db.Column(db.Integer, nullable=False)
    pvp_is_competitorA = db.Column(db.Float)
    pvp_is_competitorB = db.Column(db.Float)
    pvp_is_competitorA_actual = db.Column(db.Float)
    pvp_is_competitorB_actual = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('sku', 'time_key', name='_sku_time_uc'),)

@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return make_response(jsonify({"error": str(e)}), 422)

@app.route('/forecast_prices/', methods=['POST'])
def forecast_prices():
    try:
        data = ForecastInput(**request.get_json())
    except ValidationError as e:
        raise e  # Será capturado pelo errorhandler acima
    
    # Modelo dummy - na prática você substituiria por seu modelo real
    # Aqui estamos apenas gerando valores fictícios baseados no sku e time_key
    pvp_a = hash(data.sku + str(data.time_key)) % 100 + 50  # Valor entre 50 e 150
    pvp_b = hash(data.sku + str(data.time_key + 1)) % 100 + 60  # Valor entre 60 e 160
    
    # Verificar se já existe uma entrada para este sku e time_key
    existing = PricePrediction.query.filter_by(sku=data.sku, time_key=data.time_key).first()
    
    if existing:
        return jsonify({
            "sku": existing.sku,
            "time_key": existing.time_key,
            "pvp_is_competitorA": existing.pvp_is_competitorA,
            "pvp_is_competitorB": existing.pvp_is_competitorB
        }), 200
    
    # Criar nova entrada no banco de dados
    new_prediction = PricePrediction(
        sku=data.sku,
        time_key=data.time_key,
        pvp_is_competitorA=pvp_a,
        pvp_is_competitorB=pvp_b
    )
    
    db.session.add(new_prediction)
    db.session.commit()
    
    return jsonify({
        "sku": data.sku,
        "time_key": data.time_key,
        "pvp_is_competitorA": pvp_a,
        "pvp_is_competitorB": pvp_b
    }), 201

@app.route('/actual_prices/', methods=['POST'])
def actual_prices():
    try:
        data = ActualPriceInput(**request.get_json())
    except ValidationError as e:
        raise e  # Será capturado pelo errorhandler acima
    
    # Buscar previsão existente
    prediction = PricePrediction.query.filter_by(sku=data.sku, time_key=data.time_key).first()
    
    if not prediction:
        return jsonify({"error": "No forecast found for this sku and time_key"}), 422
    
    # Atualizar com os valores reais
    prediction.pvp_is_competitorA_actual = data.pvp_is_competitorA_actual
    prediction.pvp_is_competitorB_actual = data.pvp_is_competitorB_actual
    db.session.commit()
    
    return jsonify({
        "sku": prediction.sku,
        "time_key": prediction.time_key,
        "pvp_is_competitorA": prediction.pvp_is_competitorA,
        "pvp_is_competitorB": prediction.pvp_is_competitorB,
        "pvp_is_competitorA_actual": prediction.pvp_is_competitorA_actual,
        "pvp_is_competitorB_actual": prediction.pvp_is_competitorB_actual
    }), 200

@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)