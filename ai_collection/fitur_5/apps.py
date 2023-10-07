from django.apps import AppConfig
from pathlib import Path
import joblib
import os

class Fitur5Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_5'

    besttime_to_bill_model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_nb_besttime.joblib')
    besttime_to_bill_model = joblib.load(besttime_to_bill_model_path)
