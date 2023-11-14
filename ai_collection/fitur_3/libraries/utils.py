"""
This module provides methods for appending new data to a debtor dataset.

It includes functions to append new debtor information and prediction results to the dataset file.
The primary function, 'append_dataset_with_new_data', takes the name of the dataset file,
debtor information, and prediction results as input and appends a new row to the dataset.

Functions:
- append_dataset_with_new_data(dataset_file_name, data, result): Appends new data and result
  to the specified dataset file.
- load_dataset_path(filename): Loads the full path of the dataset file.
- build_new_row(data, result): Builds a new row for the debtor dataset by combining debtor
  information and prediction results.
- append_new_row(dataset_path, new_row): Appends a new row to the dataset file.
"""

import os
import pandas as pd


def append_dataset_with_new_data(dataset_file_name, data, result):
    """
    Append new data and result to the specified dataset file.

    Parameters:
    - dataset_file_name (str): The name of the dataset file to which new data will be appended.
    - data (dict): A dictionary containing information about the debtor.
    - result (dict): A dictionary containing prediction results.

    Returns:
    - None

    Side Effects:
    - Appends a new row with debtor information and prediction results to the dataset file.
    """
    dataset_path = load_dataset_path(dataset_file_name)
    new_row = build_new_row(data, result)
    append_new_row(dataset_path, new_row)

def load_dataset_path(filename):
    """
    Load the full path of the dataset file.

    Parameters:
    - filename (str): The name of the dataset file.

    Returns:
    - str: The full path of the dataset file.
    """
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(feature_dir, "..", "dataset")
    file_path = os.path.join(dataset_dir, filename)

    return file_path

def build_new_row(data, result):
    """
    Build a new row for the debtor dataset by combining debtor information and prediction results.

    Parameters:
    - data (dict): A dictionary containing information about the debtor.
    - result (dict): A dictionary containing prediction results.

    Returns:
    - pd.DataFrame: A new row for the dataset with debtor information and prediction results.
    """
    input_df = pd.DataFrame({key: [value] for key, value in data.items()})

    input_df_first_part = input_df[['debtor_nik', 'debtor_name',
                                    'debtor_gender', 'debtor_birth_place',
                                    'debtor_age', 'debtor_address', 'debtor_zip',
                                    'debtor_rt', 'debtor_rw', 'debtor_marital_status',
                                    'debtor_occupation', 'debtor_company',
                                    'debtor_number', 'collection_day_type']]
    # Insert best_collection_time column here
    input_df_second_part = input_df[['loan_amount', 'debtor_education_level',
                                     'credit_score', 'aging',
                                     'previous_collection_status',
                                     'previous_payment_status',
                                     'amount_of_late_days', 'tenure']]
    #Insert best_collection_method column here
    input_df_third_part = input_df[['debtor_latitude', 'debtor_longitude']]

    result_best_collection_time_df = pd.DataFrame(
        {key: [result[key]] for key in ['best_collection_time']}
    )
    result_best_collection_method_df = pd.DataFrame(
        {key: [result[key]] for key in ['best_collection_method']}
    )

    new_row = pd.concat([input_df_first_part,
                         result_best_collection_time_df,
                         input_df_second_part,
                         result_best_collection_method_df,
                         input_df_third_part], axis=1)

    return new_row

def append_new_row(dataset_path, new_row):
    """
    Append a new row to the dataset file.

    Parameters:
    - dataset_path (str): The full path of the dataset file.
    - new_row (pd.DataFrame): The new row to be appended to the dataset.

    Returns:
    - None

    Side Effects:
    - Appends a new row to the dataset file.
    """
    new_row.to_csv(dataset_path, mode='a', header=False, index=False) # using append mode
