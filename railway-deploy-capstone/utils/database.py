from peewee import SqliteDatabase, Model, CharField, IntegerField, DoubleField
from playhouse.shortcuts import model_to_dict

db = SqliteDatabase('price_predictions.db')

class PricePrediction(Model):
    # ... (igual ao modelo anterior)

def save_prediction(sku, time_key, pred_A, pred_B):
    # ... (implementação para salvar no banco)

def get_prediction(sku, time_key):
    # ... (implementação para buscar previsão)

def update_actual_prices(sku, time_key, actual_A, actual_B):
    # ... (implementação para atualizar preços reais)