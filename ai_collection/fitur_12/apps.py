from django.apps import AppConfig
from pathlib import Path
import joblib
import os


class Fitur12Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_12'

    tenor_pred_model_path = os.path.join(os.path.dirname(__file__), 'model', 'rt_pipeline_05102023.joblib')
    tenor_pred_model = joblib.load(tenor_pred_model_path)

    loan_pred_model_path = os.path.join(os.path.dirname(__file__), 'model', 'rl_pipeline.joblib')
    loan_pred_model = joblib.load(loan_pred_model_path)
