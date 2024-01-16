from django.apps import AppConfig
from pathlib import Path
import joblib
import os

class Fitur20Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_20'

    risk_status_model_path = os.path.join(os.path.dirname(__file__), 'model', 'LogisticRegression_model.joblib')
    risk_status_prediction_model = joblib.load(risk_status_model_path)








