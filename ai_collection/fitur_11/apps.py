from django.apps import AppConfig
from pathlib import Path
import joblib
import os

class Fitur11Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_11'
    
    # Kredit pinjaman model path
    default_kredit_model_path = os.path.join(os.path.dirname(__file__), 'model', 'Kredit_pinjaman_v3.joblib')
    default_kredit_scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'kredit_pinjaman_scaler_v2.joblib')
    default_solution_model_path = os.path.join(os.path.dirname(__file__), 'model', 'Kredit_pinjaman_solution_v2.joblib')
    default_solution_scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'kredit_pinjaman_solution_scaler_v2.joblib')
    
    # Kredit pinjaman model
    default_kredit_model = joblib.load(default_kredit_model_path)
    default_kredit_scaler = joblib.load(default_kredit_scaler_path)
    default_solution_model = joblib.load(default_solution_model_path)
    default_solution_scaler = joblib.load(default_solution_scaler_path)
    
    # Kredit benda model path
    default_kredit_benda_model_path = os.path.join(os.path.dirname(__file__), 'model', 'Kredit_benda_v3.joblib')
    default_kredit_benda_scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'kredit_benda_scaler_v3.joblib')
    default_solution_benda_model_path = os.path.join(os.path.dirname(__file__), 'model', 'Kredit_benda_solution_v3.joblib')
    default_solution_benda_scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'kredit_benda_solution_scaler_v3.joblib')
    
    # Kredit benda model
    default_kredit_benda_model = joblib.load(default_kredit_benda_model_path)
    default_kredit_benda_scaler = joblib.load(default_kredit_benda_scaler_path)
    default_solution_benda_model = joblib.load(default_solution_benda_model_path)
    default_solution_benda_scaler = joblib.load(default_solution_benda_scaler_path)