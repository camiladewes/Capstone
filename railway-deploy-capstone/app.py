from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import sys

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///forecasts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Forecast(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sku = db.Column(db.String(50), nullable=False)
    time_key = db.Column(db.Integer, nullable=False)
    pvp_is_competitorA = db.Column(db.Float)
    pvp_is_competitorB = db.Column(db.Float)
    pvp_is_competitorA_actual = db.Column(db.Float)
    pvp_is_competitorB_actual = db.Column(db.Float)

    __table_args__ = (
        db.UniqueConstraint('sku', 'time_key', name='_sku_time_uc'),
    )

with app.app_context():
    db.create_all()

def validate_forecast_input(data):
    if not isinstance(data.get('sku'), str) or not isinstance(data.get('time_key'), int):
        return False
    return True

def validate_actual_input(data):
    required_fields = ['sku', 'time_key', 'pvp_is_competitorA_actual', 'pvp_is_competitorB_actual']
    if not all(isinstance(data.get(field), (int, float)) if 'actual' in field else isinstance(data.get(field), (str, int)) for field in required_fields):
        return False
    return True

@app.route('/forecast_prices/', methods=['POST'])
def forecast_prices():
    data = request.get_json()
    
    if not validate_forecast_input(data):
        return jsonify({"error": "Invalid input format"}), 422

    # Dummy prediction model (replace with actual model)
    dummy_prediction = {
        'pvp_is_competitorA': 100.0,  # Replace with real prediction logic
        'pvp_is_competitorB': 150.0    # Replace with real prediction logic
    }

    # Store/update forecast in database
    forecast = Forecast.query.filter_by(sku=data['sku'], time_key=data['time_key']).first()
    if not forecast:
        forecast = Forecast(
            sku=data['sku'],
            time_key=data['time_key'],
            **dummy_prediction
        )
        db.session.add(forecast)
    else:
        forecast.pvp_is_competitorA = dummy_prediction['pvp_is_competitorA']
        forecast.pvp_is_competitorB = dummy_prediction['pvp_is_competitorB']
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

    return jsonify({
        'sku': data['sku'],
        'time_key': data['time_key'],
        'pvp_is_competitorA': dummy_prediction['pvp_is_competitorA'],
        'pvp_is_competitorB': dummy_prediction['pvp_is_competitorB']
    })

@app.route('/actual_prices/', methods=['POST'])
def actual_prices():
    data = request.get_json()
    
    if not validate_actual_input(data):
        return jsonify({"error": "Invalid input format"}), 422

    forecast = Forecast.query.filter_by(
        sku=data['sku'],
        time_key=data['time_key']
    ).first()

    if not forecast:
        return jsonify({"error": "No forecast exists for this SKU and date"}), 422

    # Update actual prices
    forecast.pvp_is_competitorA_actual = data['pvp_is_competitorA_actual']
    forecast.pvp_is_competitorB_actual = data['pvp_is_competitorB_actual']
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

    return jsonify({
        'sku': forecast.sku,
        'time_key': forecast.time_key,
        'pvp_is_competitorA': forecast.pvp_is_competitorA,
        'pvp_is_competitorB': forecast.pvp_is_competitorB,
        'pvp_is_competitorA_actual': forecast.pvp_is_competitorA_actual,
        'pvp_is_competitorB_actual': forecast.pvp_is_competitorB_actual
    })


@app.route('/view_database/', methods=['GET'])
def view_database():
    forecasts = Forecast.query.all()
    return jsonify([{
        'id': f.id,
        'sku': f.sku,
        'time_key': f.time_key,
        'pvp_is_competitorA': f.pvp_is_competitorA,
        'pvp_is_competitorB': f.pvp_is_competitorB,
        'pvp_is_competitorA_actual': f.pvp_is_competitorA_actual,
        'pvp_is_competitorB_actual': f.pvp_is_competitorB_actual
    } for f in forecasts])

if __name__ == '__main__':
    app.run(debug=True)