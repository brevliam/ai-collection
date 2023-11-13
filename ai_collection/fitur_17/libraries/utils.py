"""
Module: utils.py
Description: This module contains utility functions for handling datasets.
"""

import os
import pandas as pd

def append_dataset_with_new_data(dataset_file_name, input_df, result):
    """
    Appends a new row with input and result data to the dataset.

    Args:
        dataset_file_name (str): Name of the dataset file.
        input_df (pd.DataFrame): Input data in DataFrame format.
        result (dict): Result data to be appended.

    Returns:
        None
    """
    dataset_path = load_dataset_path(dataset_file_name)
    new_row = build_new_row(input_df, result)
    append_new_row(dataset_path, new_row)

def load_dataset_path(filename):
    """
    Constructs the full path to the dataset file.

    Args:
        filename (str): Name of the dataset file.

    Returns:
        str: Full path to the dataset file.
    """
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(feature_dir, "..", "dataset")
    file_path = os.path.join(dataset_dir, filename)

    return file_path

def append_new_row(dataset_path, new_row):
    """
    Appends a new row to the dataset file.

    Args:
        dataset_path (str): Full path to the dataset file.
        new_row (pd.DataFrame): New row to be appended.

    Returns:
        None
    """
    new_row.to_csv(dataset_path, mode='a', header=False, index=False)

def build_new_row(input_df, result):
    """
    Builds a new row by combining input and result data.

    Args:
        input_df (pd.DataFrame): Input data in DataFrame format.
        result (dict): Result data.

    Returns:
        pd.DataFrame: New row combining input and result data.
    """
    result_mod_data = {key: [value] for key, value in result.items()}
    result_df = pd.DataFrame(result_mod_data)
    new_row = pd.concat([input_df, result_df], axis=1)

    return new_row
