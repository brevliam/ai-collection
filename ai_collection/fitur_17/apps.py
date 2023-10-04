from django.apps import AppConfig
import joblib
import os


class Fitur17Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_17'

    fraud_pred_model_path = os.path.join(os.path.dirname(__file__), 'model', 'fraud_model_pipeline_v03_021023.joblib')
    fraud_pred_model = joblib.load(fraud_pred_model_path)
