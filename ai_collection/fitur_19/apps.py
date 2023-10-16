from django.apps import AppConfig
from pathlib import Path
import joblib
import os
from tensorflow.keras.models import load_model

class Fitur19Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_19'

    default_payment_model_path = os.path.join(os.path.dirname(__file__), 'model', 'payment_default.joblib')
    default_payment_pred_model = joblib.load(default_payment_model_path)

    recomended_path = os.path.join(os.path.dirname(__file__), 'model', 'model_rek.h5')
    recomended_model = load_model(recomended_path)
