"""
Module for importing necessary libraries and utilities.

This module imports the 'Path' class from the 'pathlib' module, 'os' module, and 'joblib' module.
These libraries are commonly used for working with file 
paths and handling operating system-related tasks.



Note:
    Ensure that the required libraries are installed in your Python environment.
    You can install them using:
    - `pip install pathlib` for the 'pathlib' module.
    - No additional installation is needed for the 'os' and 'joblib' modules.

"""
from pathlib import Path
import os
import joblib

from django.apps import AppConfig



class Fitur4Config(AppConfig):
    """
    Configuration class for the 'fitur_4' Django app.
    Attributes:
        default_auto_field (str): The default auto field for model creation.
        name (str): The name of the Django app.
        assignment_pred_model_path (str): The file path for the assignment prediction model.
        assignment_pred_model: The loaded assignment prediction model.
    Example:
        >>> from fitur_4.apps import Fitur4Config
        >>> config = Fitur4Config()
        >>> assignment_model = config.assignment_pred_model
    See Also:
        - `assignment_pred_model` for accessing the loaded prediction model.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_4'
    assignment_pred_model_path = os.path.join(
        os.path.dirname(__file__), 'model', 'pipeline_assignment.joblib'
        )
    assignment_pred_model = joblib.load(
        assignment_pred_model_path
        )

class Fitur4Config2(AppConfig):
    """
    Configuration class for the 'fitur_4' Django app, specifically for campaign prediction.

    Attributes:
        default_auto_field (str): The default auto field for model creation.
        name (str): The name of the Django app.
        campaign_pred_model_path (str): The file path for the campaign prediction model.
        campaign_pred_model: The loaded campaign prediction model.

    Example:
        >>> from fitur_4.apps import Fitur4Config2
        >>> config = Fitur4Config2()
        >>> campaign_model = config.campaign_pred_model
    See Also:
        - `campaign_pred_model` for accessing the loaded prediction model.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitur_4'
    campaign_pred_model_path = os.path.join(
        os.path.dirname(__file__), 'model', 'pipeline_campaign.joblib'
        )
    campaign_pred_model = joblib.load(
        campaign_pred_model_path
        )
