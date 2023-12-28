"""
utils.py

This module contains utility functions for various purposes.
"""
import os
import pandas as pd

def append_dataset_with_new_data(dataset_file_name, input_df, result):
    """
    Append new data to the dataset.

    Parameters:
    - dataset_file_name (str): The name of the dataset file.
    - input_df (pd.DataFrame): The input data.
    - result (dict): The result data to append.

    Returns:
    None
    """
    dataset_path = load_dataset_path(dataset_file_name)
    new_row = build_new_row(input_df, result)
    append_new_row(dataset_path, new_row)

def load_dataset_path(filename):
    """
    Load the full path of the dataset file.

    Parameters:
    - filename (str): The name of the dataset file.

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

    Parameters:
    - dataset_path (str): The full path of the dataset file.
    - new_row (pd.DataFrame): The new row to append.

    Returns:
    None
    """
    new_row.to_csv(dataset_path, mode='a', header=False, index=False)

def build_new_row(input_df, result):
    """
    Build a new row by concatenating input data and result data.

    Parameters:
    - input_df (pd.DataFrame): The input data.
    - result (dict): The result data.

    Returns:
    pd.DataFrame: The new row.
    """
    result_mod_data = {key: [value] for key, value in result.items()}
    result_df = pd.DataFrame(result_mod_data)
    new_row = pd.concat([input_df, result_df], axis=1)
    return new_row