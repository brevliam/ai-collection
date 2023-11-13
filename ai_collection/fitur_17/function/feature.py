"""
Module: feature.py
Description: This module contains functions for predicting fraud and remedial scores.
"""

import pandas as pd
from ..apps import Fitur17Config
from ..libraries import utils

def predict_fraud_score(data):
    """
    Predicts fraud score based on the input data.

    Args:
        data (dict): Input data for prediction.

    Returns:
        dict: Result of fraud prediction.
    """
    model = Fitur17Config.fraud_pred_model
    dataset_file_name = '17_fraud_dummy_data_v01_041023.csv'

    input_df = transform_input(data)
    output = round(model.predict(input_df))
    result = transform_fraud_pred_output(output)

    utils.append_dataset_with_new_data(dataset_file_name, input_df, result)

    return result

def predict_remedial_score(data):
    """
    Predicts remedial score based on the input data.

    Args:
        data (dict): Input data for prediction.

    Returns:
        dict: Result of remedial prediction.
    """
    model = Fitur17Config.remedial_pred_model
    dataset_file_name = '17_remedial_dummy_data_v01_051023.csv'

    input_df = transform_input(data)
    output = round(model.predict(input_df))
    result = transform_remedial_pred_output(output)

    utils.append_dataset_with_new_data(dataset_file_name, input_df, result)

    return result

def transform_input(data):
    """
    Transforms input data into a DataFrame.

    Args:
        data (dict): Input data.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)

    return df

def transform_fraud_pred_output(pred):
    """
    Transforms fraud prediction output into a standardized format.

    Args:
        pred (float): Fraud prediction score.

    Returns:
        dict: Transformed fraud prediction output.
    """
    fraud_score = pred[0]
    fraud_label = ''

    if fraud_score <= 250:
        fraud_label = 'No Fraud'
    elif fraud_score >= 500:
        fraud_label = 'Fraud'
    else:
        fraud_label = 'Suspect'

    data = {
        'fraud_score': fraud_score,
        'fraud_label': fraud_label
    }

    return data

def transform_remedial_pred_output(pred):
    """
    Transforms remedial prediction output into a standardized format.

    Args:
        pred (float): Remedial prediction score.

    Returns:
        dict: Transformed remedial prediction output.
    """
    remedial_score = pred[0]
    remedial_label = ''

    if remedial_score <= 700:
        remedial_label = 'No remedial'
    else:
        remedial_label = 'remedial'

    data = {
        'remedial_score': remedial_score,
        'remedial_label': remedial_label
    }

    return data
