"""
Module: data_appender
Description: Contains methods for appending new data to the dataset.
"""

import os
import pandas as pd


def append_dataset_with_new_data(dataset_file_name, input_df, result):
    """
    Append new data to the dataset.

    Args:
        dataset_file_name (str): The filename of the dataset.
        input_df (pd.DataFrame): The input DataFrame.
        result (dict): The result data to be appended.

    Returns:
        None
    """
    dataset_path = load_dataset_path(dataset_file_name)
    new_row = build_new_row(input_df, result)
    append_new_row(dataset_path, new_row)

def load_dataset_path(filename):
    """
    Load the full path of the dataset file.

    Args:
        filename (str): The filename of the dataset.

    Returns:
        str: The full path of the dataset file.
    """
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(feature_dir, "..", "dataset")
    file_path = os.path.join(dataset_dir, filename)

    return file_path

def append_new_row(dataset_path, new_row):
    """
    Append a new row to the dataset file.

    Args:
        dataset_path (str): The full path of the dataset file.
        new_row (pd.DataFrame): The new row to be appended.

    Returns:
        None
    """
    new_row.to_csv(dataset_path, mode='a', header=False, index=False)

def build_new_row(input_df, result):
    """
    Build a new row by concatenating input DataFrame and result data.

    Args:
        input_df (pd.DataFrame): The input DataFrame.
        result (dict): The result data to be appended.

    Returns:
        pd.DataFrame: The new row.
    """
    result_mod_data = {key: [value] for key, value in result.items()}
    result_df = pd.DataFrame(result_mod_data)
    new_row = pd.concat([input_df, result_df], axis=1)

    return new_row
