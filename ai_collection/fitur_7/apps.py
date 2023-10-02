from django.apps import AppConfig
from pathlib import Path
import joblib
import os

class Fitur7Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_7'

    workload_pred_model_path = os.path.join(os.path.dirname(__file__), 'model', 'workload_pred_model_v3_231002.joblib')
    workload_pred_model = joblib.load(workload_pred_model_path)
