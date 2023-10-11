# This module contains every method for inference, as well as data transformation and preprocessing.

import pandas as pd

from ..apps import Fitur7Config
from ..libraries import utils

def predict_workload_score(data):
    model = Fitur7Config.workload_pred_model
    DATASET_FILE_NAME = 'workload_prediction_v3_230927.csv'
  
    input_df = transform_input(data)
    output = model.predict(input_df)
    result = transform_workload_pred_output(output)
    
    utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
    
    return result

def recommend_campaign(data):
    model = Fitur7Config.campaign_rec_model
    DATASET_FILE_NAME = 'workload_opt_v6_231004.csv'

    input_df = transform_input(data)
    output = model.predict(input_df)
    result = transform_campaign_rec_output(output, input_df['aging'])

    mod_result = result.copy()
    mod_result.pop('aging', None)

    utils.append_dataset_with_new_data(DATASET_FILE_NAME, 
                                        input_df, 
                                        mod_result)

    return result

def recommend_field_collector(data):
    model = Fitur7Config.field_collector_rec_model
    DATASET_FILE_NAME = 'field_collector_opt_v2_230929.csv'
    dataset_file_path = utils.load_dataset_path(DATASET_FILE_NAME)
    model.set_collector_dataset(dataset_file_path)

    output = model.recommend(data)

    return output

def transform_input(data):
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
    
    return df

def transform_workload_pred_output(pred):
    workload_score = pred[0]
    if workload_score > 1000:
        workload_score = 1000.0
    elif workload_score < 0:
        workload_score == 0.0

    workload_level = ''
    if workload_score < 500:
        workload_level = 'underload'
    elif workload_score > 700:
        workload_level = 'overload'
    else:
        workload_level = 'normal'

    data = {
        'workload_score': workload_score,
        'workload_level': workload_level
    }

    return data

def transform_campaign_rec_output(rec, aging):
    campaign_dict = {0: 'Digital',
                    1: 'Telepon',
                    2: 'Field'}

    campaign_rec = campaign_dict.get(int(rec[0]))
    data = {
        'campaign_recommendation': campaign_rec, 
        'aging':aging[0]
        }

    return data