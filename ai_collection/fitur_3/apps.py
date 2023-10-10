from django.apps import AppConfig
import os
import joblib

import pandas as pd
from keras.models import load_model

class Fitur3Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_3'

    # Load model and pipeline for best collection time prediction
    best_collection_time_model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_collection_time_model.joblib')
    best_collection_time_model = joblib.load(best_collection_time_model_path)

    best_collection_time_pipeline_path = os.path.join(os.path.dirname(__file__), 'model', 'best_collection_time_pipeline.joblib')
    best_collection_time_pipeline = joblib.load(best_collection_time_pipeline_path)

    # Load model and pipeline for best collection method prediction
    best_collection_method_model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_collection_method_model.joblib')
    best_collection_method_model = joblib.load(best_collection_method_model_path)

    best_collection_method_pipeline_path = os.path.join(os.path.dirname(__file__), 'model', 'best_collection_method_pipeline.joblib')
    best_collection_method_pipeline = joblib.load(best_collection_method_pipeline_path)

    # Load model and pipeline for best collector prediction
    best_collector_model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_collector_model.h5')
    best_collector_model = load_model(best_collector_model_path)

    best_collector_pipeline_path = os.path.join(os.path.dirname(__file__), 'model', 'best_collector_pipeline.joblib')
    best_collector_pipeline = joblib.load(best_collector_pipeline_path)

