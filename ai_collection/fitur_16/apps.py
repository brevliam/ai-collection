from django.apps import AppConfig
from pathlib import Path
import joblib
import os

class Fitur16Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_16'

    fraud_prediction_model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_modelFraud.joblib')
    fraud_prediction_model = joblib.load(fraud_prediction_model_path)

    appraisal_prediction_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model_collateral_appraisal.joblib')
    appraisal_prediction_model = joblib.load(appraisal_prediction_path )






