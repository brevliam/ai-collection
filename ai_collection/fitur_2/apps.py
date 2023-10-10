from django.apps import AppConfig
from pathlib import Path
import joblib
import os

class Fitur2Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_2'

    debtor_class_by_age_model_path = os.path.join(os.path.dirname(__file__), 'model', 'debtor_class_by_age.joblib')
    debtor_class_by_location_model_path = os.path.join(os.path.dirname(__file__), 'model', 'debtor_class_by_location.joblib')
    debtor_class_by_behavior_model_path = os.path.join(os.path.dirname(__file__), 'model', 'debtor_class_by_behavior.joblib')
    debtor_class_by_character_model_path = os.path.join(os.path.dirname(__file__), 'model', 'debtor_class_by_character.joblib')
    debtor_class_by_collector_field_model_path = os.path.join(os.path.dirname(__file__), 'model', 'debtor_class_by_collector_field.joblib')
    debtor_class_by_ses_model_path = os.path.join(os.path.dirname(__file__), 'model', 'debtor_class_by_ses.joblib')
    debtor_class_by_demography_model_path = os.path.join(os.path.dirname(__file__), 'model', 'debtor_class_by_demography.joblib')
    
    collector_class_by_age_model_path = os.path.join(os.path.dirname(__file__), 'model', 'collector_class_by_age.joblib')
    collector_class_by_location_model_path = os.path.join(os.path.dirname(__file__), 'model', 'collector_class_by_age.joblib')
    collector_class_by_behavior_model_path = os.path.join(os.path.dirname(__file__), 'model', 'collector_class_by_debtor_behavior.joblib')
    collector_class_by_character_model_path = os.path.join(os.path.dirname(__file__), 'model', 'collector_class_by_debtor_character.joblib')
    collector_class_by_collector_field_model_path = os.path.join(os.path.dirname(__file__), 'model', 'collector_class_by_collector_field.joblib')
    collector_class_by_ses_model_path = os.path.join(os.path.dirname(__file__), 'model', 'collector_class_by_ses.joblib')
    collector_class_by_demography_model_path = os.path.join(os.path.dirname(__file__), 'model', 'collector_class_by_debtor_demography.joblib')

    collector_debtor_class_by_age = joblib.load(debtor_class_by_age_model_path)
    debtor_class_by_location = joblib.load(debtor_class_by_location_model_path)
    debtor_class_by_behavior = joblib.load(debtor_class_by_behavior_model_path)
    debtor_class_by_character = joblib.load(debtor_class_by_character_model_path)
    debtor_class_by_collector_field = joblib.load(debtor_class_by_collector_field_model_path)
    debtor_class_by_ses = joblib.load(debtor_class_by_ses_model_path)
    debtor_class_by_demography = joblib.load(debtor_class_by_demography_model_path)
    
    class_by_age = joblib.load(collector_class_by_age_model_path)
    collector_class_by_location = joblib.load(collector_class_by_location_model_path)
    collector_class_by_behavior = joblib.load(collector_class_by_behavior_model_path)
    collector_class_by_character = joblib.load(collector_class_by_character_model_path)
    collector_class_by_collector_field = joblib.load(collector_class_by_collector_field_model_path)
    collector_class_by_ses = joblib.load(collector_class_by_ses_model_path)
    collector_class_by_demography = joblib.load(collector_class_by_demography_model_path)
 