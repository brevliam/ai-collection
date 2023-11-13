"""
This module contains methods for debtor and collector label prediction, 
as well as data transformation and preprocessing.
"""

import pandas as pd

from ..apps import Fitur2Config
from ..libraries import utils

def predict_debtor_label(data):
    """
    Predict debtor labels based on various models for different features.

    Parameters:
    - data (dict): Input data containing debtor information.

    Returns:
    DataFrame: Predicted labels for different features.
    """
    model1 = Fitur2Config.debtor_class_by_age
    model2 = Fitur2Config.debtor_class_by_location
    model3 = Fitur2Config.debtor_class_by_behavior
    model4 = Fitur2Config.debtor_class_by_character
    model5 = Fitur2Config.debtor_class_by_collector_field
    model6 = Fitur2Config.debtor_class_by_ses
    model7 = Fitur2Config.debtor_class_by_demography

    dataset_file_name = 'Dummy Data Debitur_v06_20231008.csv'

    input_df = transform_input(data)
    result1 = model1.predict(input_df)
    result2 = model2.predict(input_df)
    result3 = model3.predict(input_df)
    result4 = model4.predict(input_df)
    result5 = model5.predict(input_df)
    result6 = model6.predict(input_df)
    result7 = model7.predict(input_df)

    combined_results = pd.DataFrame({
          'age_label': result1,
          'location_label': result2,
          'behavior_label': result3,
          'character_label': result4,
          'collector_field_label': result5,
          'ses_label': result6,
          'demography_label': result7
    })

    utils.append_dataset_with_new_data(dataset_file_name, input_df, combined_results)

    return combined_results

def predict_collector_label(data):
    """
    Predict collector labels based on various models for different features.

    Parameters:
    - data (dict): Input data containing collector information.

    Returns:
    DataFrame: Predicted labels for different features.
    """
    model1 = Fitur2Config.collector_class_by_age
    model2 = Fitur2Config.collector_class_by_location
    model3 = Fitur2Config.collector_class_by_behavior
    model4 = Fitur2Config.collector_class_by_character
    model5 = Fitur2Config.collector_class_by_collector_field
    model6 = Fitur2Config.collector_class_by_ses
    model7 = Fitur2Config.collector_class_by_demography

    dataset_file_name = 'Dummy Data Kolektor_v04_20231008.csv'

    input_df = transform_input(data)
    result1 = model1.predict(input_df)
    result2 = model2.predict(input_df)
    result3 = model3.predict(input_df)
    result4 = model4.predict(input_df)
    result5 = model5.predict(input_df)
    result6 = model6.predict(input_df)
    result7 = model7.predict(input_df)

    combined_results = pd.DataFrame({
        'age_label': result1,
        'location_label': result2,
        'behavior_label': result3,
        'character_label': result4,
        'collector_field_label': result5,
        'ses_label': result6,
        'demography_label': result7
    })

    utils.append_dataset_with_new_data(dataset_file_name, input_df, combined_results)

    return combined_results

def transform_input(data):
    """
    Transform input data into a DataFrame.

    Parameters:
    - data (dict): Input data.

    Returns:
    DataFrame: Transformed input data as a DataFrame.
    """
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)

    return df
