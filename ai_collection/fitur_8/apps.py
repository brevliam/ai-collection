from django.apps import AppConfig
import joblib
import os


class Fitur8Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_8'

     # Load models
    Effecientcosteffectiveness_pred_model_path = os.path.join(os.path.dirname(__file__), 'model', 'adaboost_model_pipeline.joblib')
    Effecientcosteffectiveness_model = joblib.load(Effecientcosteffectiveness_pred_model_path)
