"""
Module for data processing, prediction, and saving results to a CSV file.
"""
import os
import pandas as pd

def append_dataset_next_row(dataset_file_name, input_df, result):
    """
    Save input data along with collection difficulty score and category to a CSV file.

    Parameters:
    - input_data (dict): Input data in the form of a dictionary.
    - collection_difficulty_score (float): Collection difficulty score.
    - collection_difficulty_category (str): Collection difficulty category.

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
    - result (dict): Predicted results.

    Returns:
    DataFrame: New row containing both input data and predicted results.
    """
    result_mod_data = {key: [value] for key, value in result.items()}
    result_df = pd.DataFrame(result_mod_data)
    new_row = pd.concat([input_df, result_df], axis=1)
    return new_row
 