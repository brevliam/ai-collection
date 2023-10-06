from django.apps import AppConfig
import joblib
import os

class Fitur18Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_18'
    
    # load model
    loss_reverse_path = os.path.join(os.path.dirname(__file__), 'model', 'loss_reverse.joblib')
    # credit_risk_path = os.path.join(os.path.dirname(__file__), 'model', 'credit_risk.joblib')
    # time_to_collect_path = os.path.join(os.path.dirname(__file__), 'model', '.joblib')
    # total_cost_path = os.path.join(os.path.dirname(__file__), 'model', '.joblib')
    
    loss_reverse = joblib.load(loss_reverse_path)
    # credit_risk = joblib.load(credit_risk_path)
    # time_to_collect = joblib.load(time_to_collect_path)
    # total_cost = joblib.load(total_cost_path)




