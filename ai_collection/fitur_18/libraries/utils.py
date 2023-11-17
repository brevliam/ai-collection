"""
This module contains every method for appending new data to the dataset.
"""

import os
import pandas as pd

def append_dataset_with_new_data(dataset_file_name, input_df, result):
    """
    Appends new data to the dataset.

    Parameters:
        - dataset_file_name (str): The name of the dataset file.
        - input_df (pd.DataFrame): The input DataFrame.
        - result (dict): The result to be appended to the dataset.
    """
    dataset_path = load_dataset_path(dataset_file_name)
    new_row = build_new_row(input_df, result)
    append_new_row(dataset_path, new_row)

def load_dataset_path(filename):
    """
    Loads the dataset path.

    Parameters:
        - filename (str): The name of the dataset file.

    Returns:
        str: The full path to the dataset file.
    """
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(feature_dir, "..", "dataset")
    file_path = os.path.join(dataset_dir, filename)
    return file_path

def append_new_row(dataset_path, new_row):
    """
    Appends a new row to the dataset.

    Parameters:
        - dataset_path (str): The full path to the dataset file.
        - new_row (pd.Series): The new row to be appended to the dataset.
    """
    new_row.to_csv(dataset_path, mode='a', header=False, index=False)

def build_new_row(input_df, result):
    """
    Builds a new row for the dataset.

    Parameters:
        - input_df (pd.DataFrame): The input DataFrame.
        - result (dict): The result to be added to the new row.

    Returns:
        pd.Series: The new row to be appended to the dataset.
    """
    result_mod_data = {key: [value] for key, value in result.items()}
    result_df = pd.DataFrame(result_mod_data)
    new_row = pd.concat([input_df, result_df], axis=1)

    return new_row
 