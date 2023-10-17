from django.apps import AppConfig
from pathlib import Path
import joblib
import os

class Fitur13Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_13'

    best_time_to_remind_model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_xgb_reminder825.joblib')
    best_time_to_remind_model = joblib.load(best_time_to_remind_model_path)
    
    best_time_to_follow_up_model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_xgb_follow_up80.joblib')
    best_time_to_follow_up_model = joblib.load(best_time_to_follow_up_model_path)

    reschedule_model_path = os.path.join(os.path.dirname(__file__), 'model', 'AI_Reschedule_Automation.joblib')
    reschedule_model = joblib.load(reschedule_model_path)
