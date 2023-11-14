"""
Module for appending new data to the dataset.

This module contains methods for appending new data to an existing dataset file.

Methods:
    - append_dataset_with_new_data: Appends a new row to the dataset 
    based on input data and results.
    - load_dataset_path: Returns the full path to the dataset file.
    - append_new_row: Appends a new row to the dataset file.
    - build_new_row: Builds a new row from input data and results.

"""

import os
import pandas as pd

def append_dataset_with_new_data(dataset_file_name, input_df, result):
    """
    Appends a new row to the dataset based on input data and results.

    Args:
        dataset_file_name (str): The name of the dataset file.
        input_df (pd.DataFrame): The input data as a DataFrame.
        result (dict): The result data as a dictionary.

    Returns:
        None
    """
    dataset_path = load_dataset_path(dataset_file_name)
    new_row = build_new_row(input_df, result)
    append_new_row(dataset_path, new_row)

def load_dataset_path(filename):
    """
    Returns the full path to the dataset file.

    Args:
        filename (str): The name of the dataset file.

    Returns:
        str: The full path to the dataset file.
    """
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(feature_dir, "..", "dataset")
    file_path = os.path.join(dataset_dir, filename)
    return file_path

def append_new_row(dataset_path, new_row):
    """
    Appends a new row to the dataset file.

    Args:
        dataset_path (str): The full path to the dataset file.
        new_row (pd.DataFrame): The new row to be appended to the dataset.

    Returns:
        None
    """
    new_row.to_csv(dataset_path, mode='a', header=False, index=False)

def build_new_row(input_df, result):
    """
    Builds a new row from input data and results.

    Args:
        input_df (pd.DataFrame): The input data as a DataFrame.
        result (dict): The result data as a dictionary.

    Returns:
        pd.DataFrame: The new row as a DataFrame.
    """
    result_mod_data = {key: [value] for key, value in result.items()}
    result_df = pd.DataFrame(result_mod_data)
    new_row = pd.concat([input_df, result_df], axis=1)

    return new_row
