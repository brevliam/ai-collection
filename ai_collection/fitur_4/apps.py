from django.apps import AppConfig

from pathlib import Path
import joblib
import os


class Fitur4Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_4'
    assignment_pred_model_path = os.path.join(os.path.dirname(__file__), 'model', 'pipeline_assignment.joblib')
    assignment_pred_model = joblib.load(assignment_pred_model_path)

class Fitur4Config2(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_4'
    campaign_pred_model_path = os.path.join(os.path.dirname(__file__), 'model', 'pipeline_campaign.joblib')
    campaign_pred_model = joblib.load(campaign_pred_model_path)

