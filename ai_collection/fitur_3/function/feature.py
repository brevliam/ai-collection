# This module contains every method for inference, as well as data transformation and preprocessing.

from ..apps import Fitur3Config
from ..libraries import utils
import os

import pandas as pd
import numpy as np
import math

# Main function to run all predictions
def predict_all(data):
    DATASET_FILE_NAME = 'data_debtor_ai_collection_reyhan_v02_20231004.csv'

    time = predict_best_collection_time(data)
    method = predict_best_collection_method(data)
    collector = predict_best_collector(data, time)
    summary = summarize_predictions(time, method, collector)
    
    result = combine_predictions(time, method, collector, summary)

    utils.append_dataset_with_new_data(DATASET_FILE_NAME, data, result)

    return result


# Collection of function to predict best collection time
def predict_best_collection_time(data):
    model = Fitur3Config.best_collection_time_model

    input_df = transform_input_best_collection_time(data)
    output = model.predict(input_df)
    result = transform_output_best_collection_time(output)
    
    return result

def transform_input_best_collection_time(data):
    # Input variables for the model
    required_columns = ['debtor_occupation', 'collection_day_type']

    # Use a dictionary comprehension to filter keys based on required_columns
    input_data = {key: [value] for key, value in data.items() if key in required_columns}
    df = pd.DataFrame(input_data)

    # Pipeline
    pipeline = Fitur3Config.best_collection_time_pipeline

    input_df_transformed = pd.DataFrame(pipeline.transform(df).toarray(), columns=pipeline.get_feature_names_out())
    selected_input_df_transformed = input_df_transformed[['cat__collection_day_type_Hari kerja', 'cat__debtor_occupation_Buruh', 'cat__debtor_occupation_Pegawai Negeri']]
  
    return selected_input_df_transformed

def transform_output_best_collection_time(pred):
    collection_times = {
        0: 'malam',
        1: 'pagi, malam', 
        2: 'pagi, sore, malam'
    }
    prediction = pred[0]
    best_collection_time = collection_times.get(prediction)

    data = {
        "best_collection_time": best_collection_time
    }

    return data

    
# Collections of function to predict best collection method
def predict_best_collection_method(data):
    model = Fitur3Config.best_collection_method_model

    input_df = transform_input_best_collection_method(data)
    output = model.predict(input_df)
    result = transform_output_best_collection_method(output)
    
    return result

def transform_input_best_collection_method(data):
    # Input variables for the model
    required_columns = ['aging', 'previous_collection_status', 'previous_payment_status', 'amount_of_late_days']

    # Use a dictionary comprehension to filter keys based on required_columns
    input_data = {key: [value] for key, value in data.items() if key in required_columns}
    df = pd.DataFrame(input_data)

    # Pipeline
    pipeline = Fitur3Config.best_collection_method_pipeline

    input_df_transformed = pd.DataFrame(pipeline.transform(df), columns=pipeline.get_feature_names_out())

    return input_df_transformed

def transform_output_best_collection_method(pred):
    collection_methods = {
        0: 'Field',
        1: 'Telepon', 
        2: 'WhatsApp'
    }
    prediction = pred[0]
    best_collection_method = collection_methods.get(prediction)

    data = {
        "best_collection_method": best_collection_method
    }

    return data


# Collections of function to predict best collector
def predict_best_collector(data, best_collection_time):
    model = Fitur3Config.best_collector_model

    debtor_df = transform_input_best_collector(data, best_collection_time)

    # Import collector dataset
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(feature_dir, "..", "dataset")
    file_path_collector = os.path.join(dataset_dir, 'transformed_data_collector_ai_collection.csv')
    collector_df = pd.read_csv(file_path_collector)
    collector_df = collector_df.rename(columns={'Unnamed: 0': 'collector_id'})

    # Creating debtor-collector interaction matrix
    matrix = np.zeros((len(debtor_df), len(collector_df)), dtype=int)
    matrix = pd.DataFrame(matrix)

    dt_interact = matrix.stack(dropna=True).reset_index().rename(columns={"level_0":"debtor_id", "level_1":"collector_id", 0:"y"})
    dt_interact['debtor_id'].replace([0], debtor_df['debtor_id'][0], inplace = True)
    dt_interact = dt_interact.merge(debtor_df, how="left", left_on="debtor_id", right_on="debtor_id")
    dt_interact = dt_interact.merge(collector_df, how="left", left_on="collector_id", right_on="collector_id")
    dt_interact = dt_interact.iloc[:, :-3]
    
    xdebt = dt_interact.iloc[:,3:19].values
    xcoll = dt_interact.iloc[:,19:].values
    
    # calculate distance between debtor and collectors
    lat_debtor = data.get('debtor_latitude')
    lon_debtor = data.get('debtor_longitude')
    dist = []
    
    for i in range(len(collector_df)):
        dist.append(calculate_distance(lat_debtor, lon_debtor,
                    collector_df['collector_latitude'].iloc[i],
                    collector_df['collector_longitude'].iloc[i]))
    
    # Predict similarity between debtor and collectors
    y_out = model.predict([xdebt, xcoll])
    
    y_out = pd.DataFrame(y_out)
    y_out["distance"] = dist
    result = transform_output_best_collector(y_out, collector_df)

    return result

def calculate_distance(lat1, lon1, lat2, lon2):
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
    # Input variables for the model
    required_columns = ['debtor_gender', 'debtor_age', 'debtor_education_level', 'collection_day_type']

    # Use a dictionary comprehension to filter keys based on required_columns
    input_data = {key: [value] for key, value in data.items() if key in required_columns}
    df = pd.DataFrame(input_data)

    
    # Pipeline
    pipeline = Fitur3Config.best_collector_pipeline
    
    debt_pipeline = pd.DataFrame(pipeline.transform(df).toarray(), columns=pipeline.get_feature_names_out())
    
    debtor_pipeline = pd.concat([debt_pipeline, pd.DataFrame({"best_collection_time": [value] for value in list(best_collection_time.values())})], axis=1)
    
    debtor_pipeline.drop(['cat__debtor_gender_perempuan'], axis=1, inplace=True)
    debtor_pipeline['best_collection_time_pagi'] = debtor_pipeline.apply(lambda row: 1 if 'pagi' in
                                                         row['best_collection_time'] else 0, axis=1)
    debtor_pipeline['best_collection_time_sore'] = debtor_pipeline.apply(lambda row: 1 if 'sore' in
                                                            row['best_collection_time'] else 0, axis=1)
    debtor_pipeline['best_collection_time_malam'] = debtor_pipeline.apply(lambda row: 1 if 'malam' in
                                                            row['best_collection_time'] else 0, axis=1)
    debtor_pipeline.drop(['best_collection_time'], axis=1, inplace=True)
    
    debtor_pipeline.insert(0, 'debtor_id', 0)

    return debtor_pipeline

def transform_output_best_collector(pred, collector_data):
    # Sort by highest y_out and lowest distance
    pred.sort_values(by=[0, 'distance'], ascending=[False, True], inplace=True)
  
    recommended_collector_index = pred.index[0]
    recommended_collector_name = collector_data.loc[recommended_collector_index, 'collector_name']
    recommended_collector_distance = pred.loc[recommended_collector_index, 'distance']

    recommended_collector_dict = {
        "best_collector_id": recommended_collector_index,
        "best_collector_name": recommended_collector_name,
        "best_collector_distance_to_debtor_in_km": recommended_collector_distance
    }

    return recommended_collector_dict


# A function to return a summary of all the predictions
def summarize_predictions(time, method, collector):
    best_collection_time = time.get('best_collection_time')
    best_collection_method = method.get('best_collection_method')
    best_collector_id = collector.get('best_collector_id')
    best_collector_name = collector.get('best_collector_name')
    best_collector_distance_to_debtor_in_km = collector.get('best_collector_distance_to_debtor_in_km')

    summary = "Debitur ini sebaiknya ditagih pada waktu {} dengan metode penagihan by {} oleh kolektor dengan ID: {}, nama: {}, dan jarak dengan debitur: {} km.".format(best_collection_time, best_collection_method, best_collector_id, best_collector_name, best_collector_distance_to_debtor_in_km)

    summary_dict = {
        "summary": summary
    }
    
    return summary_dict


# A function to combine all prediction results
def combine_predictions(best_collection_time, best_collection_method, best_collector, summary):
    combined_pred = best_collection_time | best_collection_method | best_collector | summary
    return combined_pred
