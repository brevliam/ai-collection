"""
Module containing methods for workload score prediction, campaign recommendation, 
and field collector recommendation.
"""

import pandas as pd

from ..apps import Fitur7Config
from ..libraries import utils

def predict_workload_score(data):
    """
    Predict workload score based on input data.

    Parameters:
    - data (dict): Input data for workload score prediction.

    Returns:
    dict: Predicted workload score and level.
    """
    model = Fitur7Config.workload_pred_model
    dataset_file_name = 'workload_prediction_v3_230927.csv'

    input_df = transform_input(data)
    output = model.predict(input_df)
    result = transform_workload_pred_output(output)

    utils.append_dataset_with_new_data(dataset_file_name, input_df, result)

    return result

def recommend_campaign(data):
    """
    Recommend a campaign based on input data.

    Parameters:
    - data (dict): Input data for campaign recommendation.

    Returns:
    dict: Recommended campaign and aging status.
    """
    model = Fitur7Config.campaign_rec_model
    dataset_file_name = 'workload_opt_v6_231004.csv'

    input_df = transform_input(data)
    output = model.predict(input_df)
    prob = model.predict_proba(input_df)
    result = transform_campaign_rec_output(output, prob, input_df['aging'])

    mod_result = result.copy()
    mod_result.pop('aging', None)

    utils.append_dataset_with_new_data(dataset_file_name, input_df, mod_result)

    return result

def recommend_field_collector(data):
    """
    Recommend a field collector based on input data.

    Parameters:
    - data (dict): Input data for field collector recommendation.

    Returns:
    dict: Recommended field collector.
    """
    model = Fitur7Config.field_collector_rec_model
    dataset_file_name = 'field_collector_opt_v2_230929.csv'
    dataset_file_path = utils.load_dataset_path(dataset_file_name)
    model.set_collector_dataset(dataset_file_path)

    output = model.recommend(data)

    return output

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

def transform_workload_pred_output(pred):
    """
    Transform workload prediction output into a standardized format.

    Parameters:
    - pred (float): Workload prediction output.

    Returns:
    dict: Transformed workload prediction result.
    """
    workload_score = pred[0]
    if workload_score > 1000:
        workload_score = 1000.0
    elif workload_score < 0:
        workload_score = 0.0

    workload_level = ''
    if workload_score < 500:
        workload_level = 'underload'
    elif workload_score > 700:
        workload_level = 'overload'
    else:
        workload_level = 'normal'

    rounded_workload_score = round(workload_score, 2)

    data = {
        'workload_score': rounded_workload_score,
        'workload_level': workload_level
    }

    return data

def transform_campaign_rec_output(rec, prob, aging):
    """
    Transform campaign recommendation output into a standardized format.

    Parameters:
    - rec (float): Campaign recommendation output.
    - aging (str): Aging status.

    Returns:
    dict: Transformed campaign recommendation result.
    """
    campaign_dict = {0: 'Digital',
                    1: 'Telepon',
                    2: 'Field'}

    aging = aging[0]

    if aging == 'Lancar':
        aging = 'Lancar: Tidak ada tunggakan'
    elif aging == 'DPK':
        aging = 'DPK (Dalam Perhatian Khusus): Tunggakan 1-90 hari'
    elif aging == 'Kurang Lancar':
        aging = 'Kurang Lancar: Tunggakan 91-120 hari'
    elif aging == 'Diragukan':
        aging = 'Diragukan: Tunggakan 121-180 hari'
    elif aging == 'Macet':
        aging = 'Macet: Tunggakan lebih dari 180 hari'

    prob_rec = prob[0][int(rec[0])]
    prob_rec = round(prob_rec, 2)

    campaign_rec = campaign_dict.get(int(rec[0]))
    next_campaign = ''
    if campaign_rec == 'Digital':
        next_campaign = 'Telepon'
    elif campaign_rec == 'Telepon':
        next_campaign = 'Field'
    else:
        next_campaign = 'Field'

    data = {
        'campaign_recommendation': campaign_rec,
        'probability': prob_rec, 
        'next_campaign': next_campaign,
        'aging': aging
    }

    return data
