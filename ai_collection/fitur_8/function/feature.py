"""
This module contains methods for inference, data transformation, and preprocessing.
"""

import pandas as pd
from ..apps import Fitur8Config
from ..libraries import utils


DATASET_FILE_NAME = 'AIcollection_costeffectiveness_terbaruu.csv'


def predict_efficient_human_resources_score(data):
    """
    Predicts the Efficient Human Resources score and updates the dataset file.

    Parameters:
    - data (dict): Input data for prediction.

    Returns:
    dict: Result of the prediction with the score and label.
    """
    model = Fitur8Config.Effecientcosteffectiveness_model
    input_df = transform_input(data)
    output = model.predict(input_df)
    result = transform_effecientcosteffectiveness_pred_output(output)
    utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
    return result


def transform_input(data):
    """
    Transforms input data into a Pandas DataFrame.

    Parameters:
    - data (dict): Input data.

    Returns:
    pd.DataFrame: Transformed DataFrame.
    """
    data = {key: [value] for key, value in data.items()}
    data_frame = pd.DataFrame(data)
    return data_frame


def transform_effecientcosteffectiveness_pred_output(pred):
    """
    Transforms the prediction output into a dictionary with score and label.

    Parameters:
    - pred: Prediction output.

    Returns:
    dict: Transformed data with the score and label.
    """
    effecient_human_resources_score = pred[0]
    effecient_human_resources_label = ''

    if effecient_human_resources_score == 0:
        effecient_human_resources_label = 'Efficiency & Effectiveness'
    elif effecient_human_resources_score >= 3:
        effecient_human_resources_label = 'Efficiency & Effectiveness'
    else:
        effecient_human_resources_label = 'NO Efficiency & Effectiveness'

    data = {
        'Effecientcosteffectiveness_score': effecient_human_resources_score,
        'Effecientcosteffectiveness_label': effecient_human_resources_label
    }
    return data
