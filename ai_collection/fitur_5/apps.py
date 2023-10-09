from django.apps import AppConfig
from pathlib import Path
from tensorflow.keras.models import load_model
import joblib
import os

class Fitur5Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_5'

    besttime_to_bill_model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_nb_besttime.joblib')
    besttime_to_bill_model = joblib.load(besttime_to_bill_model_path)

    recsys_collector_assignments_model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_as.h5')
    recsys_collector_assignments_model = load_model(recsys_collector_assignments_model_path)

    interaction_eficiency_model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_cluster.joblib')
    interaction_eficiency_model = joblib.load(interaction_eficiency_model_path)
