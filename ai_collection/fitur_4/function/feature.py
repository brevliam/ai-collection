"""
Module for importing necessary libraries and configurations.

This module imports required libraries and configurations for use in other parts of the application.
It includes imports for NumPy, Pandas, and configurations from the 'apps' and 'libraries' modules.

Example:
    import numpy as np
    import pandas as pd
    from ..apps import Fitur4Config, Fitur4Config2
    from ..libraries import utils

Note:
    Ensure that the relative import paths (e.g., '..apps', '..libraries') are correctly configured
    based on your project structure.

See Also:
    - `Fitur4Config` and `Fitur4Config2` for application configurations.
    - `utils` module from the 'libraries' package for utility functions.
"""
import numpy as np
import pandas as pd
from ..apps import Fitur4Config, Fitur4Config2
from ..libraries import utils


def predict_assignment(data):
    """
    Predicts an assignment using the provided data and updates the dataset.

    Parameters:
        data (dict): Input data for prediction.

    Returns:
        str: The predicted assignment.


    Note:
        - Ensure that the 'assignment_pred_model' is properly configured in Fitur4Config.
        - The input data should be in a dictionary format.

    Example:
        >>> data = {'feature1': value1, 'feature2': value2, ...}
        >>> assignment = predict_assignment(data)
    """
    model = Fitur4Config.assignment_pred_model
    dataset_file_name = 'df_assignment.csv'
    input_df = transform_input(data)

    output = model.predict(input_df)
    output_df = output  # Consider removing if output_model is not used
    utils.append_dataset_with_new_data_assignment(dataset_file_name, input_df, output_df)
    return output[0]


def predict_campaign(data):
    """
    Predicts a campaign based on the provided data and updates the dataset.

    Parameters:
        data (dict): Input data for prediction.

    Returns:
        str: Campaign cluster

    Note:
        - Ensure that the 'campaign_pred_model' is properly configured in Fitur4Config2.
        - The input data should be in a dictionary format.
        - The 'Pred_cluster_campaign' function is used to determine the campaign Label.

    Example:
        >>> data = {'feature1': value1, 'feature2': value2, ...}
        >>> campaign = predict_campaign(data)
    """
    model = Fitur4Config2.campaign_pred_model
    dataset_file_name = 'df_campaign.csv'
    input_df = transform_input(data)
    output = model.predict(input_df)
    output_model = pred_cluster_campaign(output, input_df)
    output_df = np.array([output_model])
    utils.append_dataset_with_new_data_camapign(dataset_file_name, input_df, output_df)
    return output_model


def transform_input(data):
    """
    Transform input data into a DataFrame.

    Parameters:
        data (dict): Input data in dictionary format.

    Returns:
        pd.DataFrame: Transformed data in DataFrame format.

    Example:
        >>> data = {'feature1': value1, 'feature2': value2, ...}
        >>> transformed_df = transform_input(data)
    """
    df = pd.DataFrame([data])
    return df


def pred_cluster_campaign(predictions, data):
    """
    Determine the campaign based on predictions and input data.

    Parameters:
        predictions (array-like): Model predictions.
        data (pd.DataFrame): Input data for making campaign predictions.

    Returns:
        str: The selected campaign based on the predictions.

    Example:
        >>> predictions = [0]
        >>> data = pd.DataFrame({'total_visit': [5], 'foreclosure': [True]})
        >>> campaign = Pred_cluster_campaign(predictions, data)
    """
    var_vis = 'total_visit'
    foreclosure = 'foreclosure'
    visit_campaign_cont = 2
    visit_campaign_beg = 0
    call_campaign = 1
    predictions = predictions[0]
    campaign_selected = None
    if predictions == visit_campaign_cont:
        if data[var_vis].values[0] > 4:
            campaign_selected = 'Lakukan kunjungan Ulang'
        else:
            if data[foreclosure].values[0] is False:
                campaign_selected = 'Lakukan Penyitaan Barang'
            elif data[foreclosure].values[0] is True:
                campaign_selected = 'Lakukan Penutupan buku Kreditur'
    elif predictions == visit_campaign_beg:
        campaign_selected = 'Lakukan kunjungan Pertama'
    elif predictions == call_campaign:
        campaign_selected = 'Lakukan Telfon'
    if campaign_selected is None:
        campaign_selected = 'No campaign selected'
    return campaign_selected
