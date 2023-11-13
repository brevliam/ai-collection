"""Module for data preprocessing and prediction using the collection difficulty model."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from ..apps import Fitur1Config

# List of fields to be excluded from the input data
unwanted_fields = [
    'debtor_nik', 'debtor_name', 'debtor_gender', 'debtor_birth_place',
    'debtor_age', 'debtor_zip', 'debtor_rt', 'debtor_rw', 'debtor_address',
    'debtor_number', 'bulan_1', 'bulan_2', 'bulan_3', 'bulan_4', 'bulan_5',
    'bulan_6', 'bulan_7', 'bulan_8', 'bulan_9', 'bulan_10', 'bulan_11',
    'bulan_12', 'occupation', 'financial_situation', 'delay_history',
    'communication_channel', 'working_time', 'payment_patterns',
    'payment_method'
]

def encode_categorical_data(input_df):
    """
    Encode categorical columns using Label Encoding.

    Parameters:
    - input_df (DataFrame): Input data containing categorical columns.

    Returns:
    DataFrame: Transformed DataFrame with encoded categorical columns.
    """
    label_encoder = LabelEncoder()

    # Encode specified categorical columns
    categorical_columns = [
        'debtor_gender',
        'communication_channel',
        'working_time',
        'payment_patterns',
        'payment_method'
    ]

    for column in categorical_columns:
        input_df[column] = label_encoder.fit_transform(input_df[column])

    return input_df

class HomeOwnershipTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for scaling the 'home_ownership' column using MinMaxScaler.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, x):
        """
        Fit the MinMaxScaler on the 'home_ownership' column.

        Parameters:
        - x (DataFrame): Input data.

        Returns:
        self: The fitted transformer object.
        """
        self.scaler.fit(x[["home_ownership"]])
        return self

    def transform(self, x):
        """
        Transform the 'home_ownership' column using the fitted scaler.

        Parameters:
        - x (DataFrame): Input data.

        Returns:
        DataFrame: Transformed DataFrame with scaled 'home_ownership' column.
        """
        x["home_ownership"] = x["home_ownership"].astype(int)
        x[["home_ownership"]] = self.scaler.fit_transform(x[["home_ownership"]])
        return x

class EmploymentStatusTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for scaling the 'employment_status' column using MinMaxScaler
    and mapping categorical values to numerical representations.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, x):
        """
        Fit the MinMaxScaler on the 'employment_status' column.

        Parameters:
        - x (DataFrame): Input data.

        Returns:
        self: The fitted transformer object.
        """
        self.scaler.fit(x[["employment_status"]])
        return self

    def transform(self, x):
        """
        Transform the 'employment_status' column using the fitted scaler
        and map categorical values to numerical representations.

        Parameters:
        - x (DataFrame): Input data.

        Returns:
        DataFrame: Transformed DataFrame with scaled 'employment_status' column.
        """
        employment_status_dict = {
            'Employed': 0,
            'Full-time': 1,
            'Self-employed': 2,
            'Not available': 3,
            'Other': 4,
            'Part-time': 5,
            'Not employed': 6,
            'Retired': 7
        }
        x["employment_status"] = x["employment_status"].replace(employment_status_dict)
        x[["employment_status"]] = self.scaler.fit_transform(x[["employment_status"]])
        return x

def determine_score_category(collection_difficulty_score):
    """
    Determine the collection difficulty category based on the given score.

    Parameters:
    - collection_difficulty_score (float): Collection difficulty score.

    Returns:
    str: Collection difficulty category.
    """
    if collection_difficulty_score >= 0 and collection_difficulty_score <= 281:
        collection_difficulty_category = "mudah"
    elif collection_difficulty_score >= 282 and collection_difficulty_score <= 372:
        collection_difficulty_category = "sedang"
    elif collection_difficulty_score >= 373 and collection_difficulty_score <= 1000:
        collection_difficulty_category = "sulit"
    else:
        collection_difficulty_category = "Skor Melebihi Parameter"
    return collection_difficulty_category

# Initialization of the data preprocessing pipeline
data_pipeline = Pipeline([
    ('home_ownership_transformer', HomeOwnershipTransformer()),
    ('employment_status_transformer', EmploymentStatusTransformer()),
])

def process_input_data(input_data):
    """
    Process input data by encoding categorical columns and applying the data pipeline.

    Parameters:
    - input_data (dict): Input data in the form of a dictionary.

    Returns:
    numpy.ndarray: Processed input data as a numpy array.
    """
    # Serialize the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical columns using label encoding
    input_df = encode_categorical_data(input_df)

    # Drop unwanted columns
    input_df = input_df.drop(columns=unwanted_fields)

    # Use the data pipeline
    processed_data = data_pipeline.transform(input_df)

    # Convert the processed data to a numpy array
    input_array = processed_data.values

    return input_array

def process_and_predict(input_data):
    """
    Process input data and make predictions using the collection difficulty model.

    Parameters:
    - input_data (dict): Input data in the form of a dictionary.

    Returns:
    tuple: Tuple containing processed input array, collection difficulty score, and category.
    """
    # Process input data
    input_array = process_input_data(input_data)

    # Predict and get the category
    score, category = predict_collection_difficulty(input_array)

    return input_array, score, category

def predict_collection_difficulty(input_array):
    """
    Predict collection difficulty score and determine the category.

    Parameters:
    - input_array (numpy.ndarray): Processed input data as a numpy array.

    Returns:
    tuple: Tuple containing collection difficulty score and category.
    """
    # Perform prediction using the model
    collection_difficulty_score = Fitur1Config.model.predict(input_array)[0]

    # Determine the prediction result based on criteria
    collection_difficulty_category = determine_score_category(collection_difficulty_score)

    return collection_difficulty_score, collection_difficulty_category
