from django.apps import AppConfig
from pathlib import Path
import joblib
import os

class Fitur9Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_9'
    
    default_solution_model_path = os.path.join(os.path.dirname(__file__), 'model', 'kredit_recommendation_solution_v4.joblib.joblib')
    default_solution_scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'kredit_recommendation_solution_scaler_v4.joblib.joblib')
    
    default_solution_model = joblib.load(default_solution_model_path)
    default_solution_scaler = joblib.load(default_solution_scaler_path)
    