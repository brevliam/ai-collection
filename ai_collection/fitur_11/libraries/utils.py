"""This module contains every method for appending new data to the dataset."""
import os

def load_dataset_path(filename):
    """
    Return the full file path for the given dataset filename.

    Parameters:
    - filename (str): The name of the dataset file.

    Returns:
    - str: The full file path.
    """
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(feature_dir, "..", "dataset")
    file_path = os.path.join(dataset_dir, filename)

    return file_path

def append_new_row(dataset_file_name, data_frame):
    """
    Append a new row to the dataset.

    Parameters:
    - dataset_file_name (str): The name of the dataset file.
    - data_frame (pd.DataFrame): The DataFrame containing the row to be appended.

    Returns:
    - None
    """
    dataset_path = load_dataset_path(dataset_file_name)
    with open(dataset_path, 'a', newline='', encoding='utf-8') as file:
        data_frame.to_csv(file, header=False, index=False)
