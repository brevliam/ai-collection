"""
This module contains methods for appending new data to the dataset.
"""

import os
import pandas as pd

def append_dataset_with_new_data(dataset_file_name, input_df, result):
    """
    Append new data along with predictions to the dataset.

    Parameters:
    - dataset_file_name (str): Name of the dataset file.
    - input_df (DataFrame): Input data as a DataFrame.
    - result (DataFrame): Predicted results as a DataFrame.

    Returns:
    None
    """
    dataset_path = load_dataset_path(dataset_file_name)
    new_row = build_new_row(input_df, result)
    append_new_row(dataset_path, new_row)

def load_dataset_path(filename):
    """
    Load the full path to the dataset file.

    Parameters:
    - filename (str): Name of the dataset file.

    Returns:
    str: Full path to the dataset file.
    """
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(feature_dir, "..", "dataset")
    file_path = os.path.join(dataset_dir, filename)

    return file_path

def append_new_row(dataset_path, new_row):
    """
    Append a new row to the dataset file.

    Parameters:
    - dataset_path (str): Full path to the dataset file.
    - new_row (DataFrame): New row to be appended.

    Returns:
    None
    """
    new_row.to_csv(dataset_path, mode='a', header=False, index=False)

def build_new_row(input_df, result):
    """
    Build a new row by concatenating input data and predicted results.

    Parameters:
    - input_df (DataFrame): Input data as a DataFrame.
    - result (DataFrame): Predicted results as a DataFrame.

    Returns:
    DataFrame: New row containing both input data and predicted results.
    """
    result_df = pd.DataFrame(result)
    new_row = pd.concat([input_df, result_df], axis=1)

    return new_row