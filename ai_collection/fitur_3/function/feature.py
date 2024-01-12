"""
This module provides functions for predicting the best collection strategy for debtors
in an AI collection system. It includes functions for predicting the best collection time,
method, and collector, as well as functions to transform the input data and the prediction
output of the models.

Dependencies:
- os
- math
- pandas as pd
- numpy as np
- Fitur3Config from ..apps
- utils from ..libraries

Functions:
- predict_all(data)
- predict_best_collection_time(data)
- transform_input_best_collection_time(data)
- transform_output_best_collection_time(pred)
- predict_best_collection_method(data)
- transform_input_best_collection_method(data)
- transform_output_best_collection_method(pred)
- predict_best_collector(data, best_collection_time)
- calculate_distance(lat1, lon1, lat2, lon2)
- transform_input_best_collector(data, best_collection_time)
- transform_output_best_collector(pred)
- summarize_predictions(time, method, collector)
- combine_predictions(best_collection_time, best_collection_method, best_collector, summary)

Variables:
- BEST_TIME_MODEL
- BEST_METHOD_MODEL
- BEST_COLLECTOR_MODEL
- MODULE_DIR
- DATASET_DIR
- FILE_PATH_COLLECTOR_DATA
- COLLECTOR_DF
- DATASET_FILE_NAME
"""


import os
import math

import pandas as pd
import numpy as np

from ..apps import Fitur3Config
from ..libraries import utils

BEST_TIME_MODEL = Fitur3Config.best_collection_time_model
BEST_METHOD_MODEL = Fitur3Config.best_collection_method_model
BEST_COLLECTOR_MODEL = Fitur3Config.best_collector_model

# Import collector dataset
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(MODULE_DIR, "..", "dataset")
FILE_PATH_COLLECTOR_DATA = os.path.join(DATASET_DIR, 'transformed_data_collector_ai_collection_strategy_v2.csv')
COLLECTOR_DF = pd.read_csv(FILE_PATH_COLLECTOR_DATA)
COLLECTOR_DF = COLLECTOR_DF.rename(columns={'Unnamed: 0': 'collector_id'})

DATASET_FILE_NAME = 'data_debtor_ai_collection_strategy_v03_20231229.csv'


def predict_all(data):
    """
    Run all predictions for debtor AI collection strategy.

    Parameters:
    - data (dict): Input data containing information about the debtor for prediction.

    Returns:
    - dict: A dictionary containing the combined predictions and summary.

    Side Effects:
    - Appends the result and new data to the dataset file specified by DATASET_FILE_NAME.
    """
    time = predict_best_collection_time(data)
    method = predict_best_collection_method(data)
    collector = predict_best_collector(data, time)
    summary = summarize_predictions(time, method, collector)
    result = combine_predictions(time, method, collector, summary)

    utils.append_dataset_with_new_data(DATASET_FILE_NAME, data, result)

    return result


def predict_best_collection_time(data):
    """
    Predict the best collection time for a debtor.

    Parameters:
    - data (dict): Input data containing debtor information for best collection strategy prediction.

    Returns:
    - dict: A dictionary containing the predicted best collection time.
    """
    input_df = transform_input_best_collection_time(data)
    output = BEST_TIME_MODEL.predict(input_df)
    result = transform_output_best_collection_time(output)

    return result


def transform_input_best_collection_time(data):
    """
    Transform input data for predicting the best collection time.

    Parameters:
    - data (dict): Input data containing debtor information for best collection strategy prediction.

    Returns:
    - pandas.DataFrame: Transformed input DataFrame with selected features for predicting the best
      collection time.
    """
    # Input variables for the model
    # required_columns = ['debtor_occupation', 'collection_day_type']

    # Use a dictionary comprehension to filter keys based on required_columns
    input_data = {key: [value] for key, value in data.items()}

    df = pd.DataFrame(input_data)

    # Pipeline
    pipeline = Fitur3Config.best_collection_time_pipeline

    input_df_transformed = pd.DataFrame(pipeline.transform(df).toarray(),
                                        columns=pipeline.get_feature_names_out())
    selected_input_df_transformed = input_df_transformed[['cat__collection_day_type_Hari kerja',
                                                          'cat__debtor_occupation_group_Buruh',
                                                          'cat__debtor_occupation_group_Pengusaha']]

    return selected_input_df_transformed


def transform_output_best_collection_time(pred):
    """
    Transform the output of best collection time prediction.

    Parameters:
    - pred (numpy.ndarray): Output prediction array from the model.

    Returns:
    - dict: Transformed data containing the output of best collection time prediction.
    """
    collection_times = {
        0: 'malam',
        1: 'pagi, malam',
        2: 'pagi, sore, malam'
    }
    prediction = pred[0]
    best_collection_time = collection_times.get(prediction)

    best_time_dict = {
        "best_collection_time": best_collection_time
    }

    return best_time_dict


def predict_best_collection_method(data):
    """
    Predict the best collection method for a debtor.

    Parameters:
    - data (dict): Input data containing debtor information for best collection strategy prediction.

    Returns:
    - dict: A dictionary containing the predicted best collection method.
    """
    input_df = transform_input_best_collection_method(data)
    output = BEST_METHOD_MODEL.predict(input_df)
    result = transform_output_best_collection_method(output)

    return result


def transform_input_best_collection_method(data):
    """
    Transform input data for predicting the best collection method.

    Parameters:
    - data (dict): Input data containing debtor information for best collection strategy prediction.

    Returns:
    - pd.DataFrame: Transformed input data ready for predicting the best collection method.
    """
    # Input variables for the model
    # required_columns = ['aging', 'previous_collection_status',
    #                    'previous_payment_status',
    #                   'amount_of_late_days']

    # Use a dictionary comprehension to filter keys based on required_columns
    input_data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(input_data)

    # Pipeline
    pipeline = Fitur3Config.best_collection_method_pipeline

    input_df_transformed = pd.DataFrame(pipeline.transform(df),
                                        columns=pipeline.get_feature_names_out())

    return input_df_transformed


def transform_output_best_collection_method(pred):
    """
    Transform the output of best collection method prediction.

    Parameters:
    - pred (numpy.ndarray): Output prediction array from the model.

    Returns:
    - dict: Transformed data containing the output of best collection method prediction.
    """
    collection_methods = {
        0: 'Field',
        1: 'Telepon',
        2: 'WhatsApp'
    }
    prediction = pred[0]
    best_collection_method = collection_methods.get(prediction)

    best_method_dict = {
        "best_collection_method": best_collection_method
    }

    return best_method_dict


def predict_best_collector(data, best_collection_time):
    """
    Predict the best collector based on input data and the predicted best collection time.

    This function takes input data, the predicted best collection time,
    and predicts the best collector using a collector dataset.
    It calculates the distance between the debtor and collectors, predicts the similarity,
    and returns information about the recommended collector.

    Parameters:
    - data (dict): Input data containing debtor information for best collection strategy prediction.
    - best_collection_time (dict): The predicted best collection time for the input data.

    Returns:
    - dict: Information about the recommended collector, including collector ID, name,
            and distance to the debtor.
    """
    debtor_df = transform_input_best_collector(data, best_collection_time)

    # Creating debtor-collector interaction matrix
    matrix = np.zeros((len(debtor_df), len(COLLECTOR_DF)), dtype=int)
    matrix = pd.DataFrame(matrix)

    dt_interact = (
        matrix
        .stack(dropna=True)
        .reset_index()
        .rename(columns={"level_0": "debtor_id", "level_1": "collector_id", 0: "y"})
    )
    dt_interact['debtor_id'].replace([0], debtor_df['debtor_id'][0], inplace = True)
    dt_interact = dt_interact.merge(debtor_df, how="left",
                                    left_on="debtor_id",
                                    right_on="debtor_id")
    dt_interact = dt_interact.merge(COLLECTOR_DF,
                                    how="left",
                                    left_on="collector_id",
                                    right_on="collector_id")
    dt_interact = dt_interact.iloc[:, :-3]

    xdebt = dt_interact.iloc[:,3:22].values
    xcoll = dt_interact.iloc[:,22:].values

    # calculate distance between debtor and collectors
    lat_debtor = data.get('debtor_latitude')
    lon_debtor = data.get('debtor_longitude')
    dist = []

    for i in range(len(COLLECTOR_DF)):
        dist.append(calculate_distance(lat_debtor, lon_debtor,
                    COLLECTOR_DF['collector_latitude'].iloc[i],
                    COLLECTOR_DF['collector_longitude'].iloc[i]))

    # Predict similarity between debtor and collectors
    y_out = BEST_COLLECTOR_MODEL.predict([xdebt, xcoll])

    y_out = pd.DataFrame(y_out)
    y_out["distance"] = dist
    result = transform_output_best_collector(y_out)

    return result


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface using latitude and longitude.

    Parameters:
    - lat1 (float): Latitude of the first point.
    - lon1 (float): Longitude of the first point.
    - lat2 (float): Latitude of the second point.
    - lon2 (float): Longitude of the second point.

    Returns:
    - float: Distance between the two points in kilometers.
    """
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    radius = 6371  # Radius of the Earth in kilometers
    distance = radius * c
    # return distance between the two points in kilometers.
    return distance


def transform_input_best_collector(data, best_collection_time):
    """
    Transform input data for best collector prediction.

    Parameters:
    - data (dict): Input data containing debtor information for best collection strategy prediction.
    - best_collection_time (dict): The predicted best collection time for the input data.

    Returns:
    - pd.DataFrame: Transformed input data for predicting the best collector.
    """
    # Input variables for the model
    required_columns = ['debtor_gender',
                        'debtor_age',
                        'debtor_education_level',
                        'collection_day_type']

    # Use a dictionary comprehension to filter keys based on required_columns
    input_data = {key: [value] for key, value in data.items() if key in required_columns}
    df = pd.DataFrame(input_data)

    # Pipeline
    pipeline = Fitur3Config.best_collector_pipeline

    debt_pipeline = pd.DataFrame(pipeline.transform(df).toarray(),
                                 columns=pipeline.get_feature_names_out())

    best_time_df = pd.DataFrame(
        {"best_collection_time": [value] for value in list(best_collection_time.values())}
    )
    debtor_pipeline = pd.concat([debt_pipeline, best_time_df], axis=1)

    debtor_pipeline.drop(['cat__debtor_gender_perempuan'], axis=1, inplace=True)
    debtor_pipeline['best_collection_time_pagi'] = debtor_pipeline.apply(lambda row: 1 if 'pagi' in
                                                        row['best_collection_time'] else 0, axis=1)
    debtor_pipeline['best_collection_time_sore'] = debtor_pipeline.apply(lambda row: 1 if 'sore' in
                                                        row['best_collection_time'] else 0, axis=1)
    debtor_pipeline['best_collection_time_malam'] = (
        debtor_pipeline
        .apply(lambda row: 1 if 'malam' in row['best_collection_time'] else 0, axis=1)
    )
    debtor_pipeline.drop(['best_collection_time'], axis=1, inplace=True)
    debtor_pipeline.insert(0, 'debtor_id', 0)

    return debtor_pipeline


def transform_output_best_collector(pred):
    """
    Transform the output of the best collector prediction.

    Parameters:
    - pred (pd.DataFrame): DataFrame containing the similarity prediction results between
                           the debtor and each collector.

    Returns:
    - dict: Dictionary containing the best collector's ID, name,
            and distance to the debtor in kilometers.
    """
    # Sort by highest y_out and lowest distance
    pred.sort_values(by=[0, 'distance'], ascending=[False, True], inplace=True)

    recommended_collector_index = pred.index[0]
    recommended_collector_name = COLLECTOR_DF.loc[recommended_collector_index, 'collector_name']
    recommended_collector_distance = pred.loc[recommended_collector_index, 'distance']

    recommended_collector_dict = {
        "best_collector_id": recommended_collector_index,
        "best_collector_name": recommended_collector_name,
        "best_collector_distance_to_debtor_in_km": recommended_collector_distance
    }

    return recommended_collector_dict


def summarize_predictions(time, method, collector):
    """
    Function to generate a summary of predictions.

    Parameters:
    - time (dict): A dictionary containing the prediction for the best collection time.
    - method (dict): A dictionary containing the prediction for the best collection method.
    - collector (dict): A dictionary containing the prediction for the best collector.

    Returns:
    - dict: A dictionary containing the summary string.
    """
    best_collection_time = time.get('best_collection_time')
    best_collection_method = method.get('best_collection_method')
    best_collector_id = collector.get('best_collector_id')
    best_collector_name = collector.get('best_collector_name')
    best_collector_distance_to_debtor_in_km = (
        collector.get('best_collector_distance_to_debtor_in_km')
    )

    predictions = [
        best_collection_time,
        best_collection_method,
        best_collector_id,
        best_collector_name,
        best_collector_distance_to_debtor_in_km
    ]

    summary = (
    f"Debitur ini sebaiknya ditagih pada waktu {predictions[0]} dengan metode penagihan "
    f"by {predictions[1]} oleh kolektor dengan ID: {predictions[2]}, nama: {predictions[3]}, "
    f"dan jarak dengan debitur: {predictions[4]:.2f} km."
    )

    summary_dict = {
        "summary": summary
    }

    return summary_dict


def combine_predictions(best_collection_time, best_collection_method, best_collector, summary):
    """
    Function to combine all prediction results and the summary into a single dictionary.

    Parameters:
    - best_collection_time (dict): A dictionary containing the prediction for
                                   the best collection time.
    - best_collection_method (dict): A dictionary containing the prediction for
                                     the best collection method.
    - best_collector (dict): A dictionary containing the prediction for the best collector.
    - summary (dict): A dictionary containing the summary string.

    Returns:
    - dict: A dictionary containing the combined prediction results.
    """
    combined_pred = best_collection_time | best_collection_method | best_collector | summary
    return combined_pred
