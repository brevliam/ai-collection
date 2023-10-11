from django.apps import AppConfig
from pathlib import Path
import joblib
import os

class Fitur19Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_19'

    default_payment_model_path = os.path.join(os.path.dirname(__file__), 'model', 'payment_default.joblib')
    default_payment_pred_model = joblib.load(default_payment_model_path)
