"""
Module for importing pandas and os modules.

Imports:
    - pandas as pd: A library for data manipulation and analysis
    - os: A module for interacting with the operating system, 
    providing functions for file and directory operations
"""
import os
import pandas as pd
def append_dataset_with_new_data_assignment(dataset_file_name, input_df, result):
    """
    Appends new data to the dataset for assignment predictions.

    Parameters:
        dataset_file_name (str): The name of the dataset file.
        input_df (pd.DataFrame): The input data in DataFrame format.
        result (dict): The result of the assignment prediction.

    Example:
        >>> dataset_file_name = 'df_assignment.csv'
        >>> input_df = pd.DataFrame({'feature1': [value1], 'feature2': [value2], ...})
        >>> result = {'collector_name': ['John Doe']}
        >>> append_dataset_with_new_data_assignment(dataset_file_name, input_df, result)

    Note:
        - The dataset file is assumed to be in a 'dataset' directory relative to this module.
        - Ensure the proper file format and structure for the dataset.

    See Also:
        - `load_dataset_path` for loading the dataset file path.
        - `build_new_row_assignment` for constructing a new row based on results.
        - `append_new_row` for appending a new row to the dataset.
    """
    dataset_path = load_dataset_path(dataset_file_name)
    new_row = build_new_row_assignment(input_df, result)
    append_new_row(dataset_path, new_row)


def append_dataset_with_new_data_campaign(dataset_file_name, input_df, result):
    """
    Appends new data to the dataset for campaign predictions.

    Parameters:
        dataset_file_name (str): The name of the dataset file.
        input_df (pd.DataFrame): The input data in DataFrame format.
        result (dict): The result of the campaign prediction.

    Example:
        >>> dataset_file_name = 'df_campaign.csv'
        >>> input_df = pd.DataFrame({'feature1': [value1], 'feature2': [value2], ...})
        >>> result = {'campaign': ['Lakukan kunjungan Pertama']}
        >>> append_dataset_with_new_data_campaign(dataset_file_name, input_df, result)

    Note:
        - The dataset file is assumed to be in a 'dataset' directory relative to this module.
        - Ensure the proper file format and structure for the dataset.

    See Also:
        - `load_dataset_path` for loading the dataset file path.
        - `build_new_row_campaign` for constructing a new row based on results.
        - `append_new_row` for appending a new row to the dataset.
    """
    dataset_path = load_dataset_path(dataset_file_name)
    new_row = build_new_row_campaign(input_df, result)
    append_new_row(dataset_path, new_row)


def load_dataset_path(filename):
    """
    Load the absolute file path for a dataset file.

    Parameters:
        filename (str): The name of the dataset file.

    Returns:
        str: The absolute file path for the dataset.

    Example:
        >>> filename = 'df_assignment.csv'
        >>> path = load_dataset_path(filename)

    Note:
        - The dataset file is assumed to be in a 'dataset' directory relative to this module.
        - Ensure the proper file format and structure for the dataset.
    """
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(feature_dir, "..", "dataset")
    file_path = os.path.join(dataset_dir, filename)
    return file_path


def append_new_row(dataset_path, new_row):
    """
    Append a new row to a dataset file.

    Parameters:
        dataset_path (str): The absolute file path for the dataset.
        new_row (pd.DataFrame): The new row to append to the dataset.

    Example:
        >>> dataset_path = '/path/to/your/dataset.csv'
        >>> new_row = pd.DataFrame({'feature1': [value1], 'feature2': [value2], ...})
        >>> append_new_row(dataset_path, new_row)

    Note:
        - Ensure the proper file format and structure for the dataset.
        - The new row is appended to the dataset without a header.

    See Also:
        - `load_dataset_path` for loading the dataset file path.
    """
    new_row.to_csv(dataset_path, mode='a', header=False, index=False)


def build_new_row_assignment(input_df, result):
    """
    Build a new row for the dataset based on assignment prediction results.

    Parameters:
        input_df (pd.DataFrame): The input data in DataFrame format.
        result (dict): The result of the assignment prediction.

    Returns:
        pd.DataFrame: The new row to append to the dataset.

    Example:
        >>> input_df = pd.DataFrame({'feature1': [value1], 'feature2': [value2], ...})
        >>> result = {'collector_name': ['John Doe']}
        >>> new_row = build_new_row_assignment(input_df, result)

    Note:
        - The 'result' dictionary is assumed to contain a 'collector_name' key.
        - The new row is constructed by concatenating the input data and result.

    See Also:
        - `append_new_row` for appending a new row to the dataset.
    """
    result_df = pd.DataFrame(result)
    result_df = result_df.rename(columns={0: 'collector_name'})
    new_row = pd.concat([input_df, result_df], axis=1)
    return new_row


def build_new_row_campaign(input_df, result):
    """
    Build a new row for the dataset based on campaign prediction results.

    Parameters:
        input_df (pd.DataFrame): The input data in DataFrame format.
        result (dict): The result of the campaign prediction.

    Returns:
        pd.DataFrame: The new row to append to the dataset.

    Example:
        >>> input_df = pd.DataFrame({'feature1': [value1], 'feature2': [value2], ...})
        >>> result = {'campaign': ['Lakukan kunjungan Pertama']}
        >>> new_row = build_new_row_campaign(input_df, result)

    Note:
        - The 'result' dictionary is assumed to contain a 'campaign' key.
        - The new row is constructed by concatenating the input data and result.

    See Also:
        - `append_new_row` for appending a new row to the dataset.
    """
    result_df = pd.DataFrame(result)
    result_df = result_df.rename(columns={0: 'campaign'})
    new_row = pd.concat([input_df, result_df], axis=1)
    return new_row
