from django.apps import AppConfig
import os
import joblib

class Fitur1Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_1'

# Create your models here.
# Load your trained model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'lasso_model_regression v2.pkl')
model = joblib.load(model_path)