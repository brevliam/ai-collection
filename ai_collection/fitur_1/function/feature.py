# feature.py

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from ..apps import model

unwanted_fields = [
    'debtor_nik', 'debtor_name', 'debtor_gender', 'debtor_birth_place',
    'debtor_age', 'debtor_zip', 'debtor_rt', 'debtor_rw', 'debtor_address',
    'debtor_number', 'bulan_1', 'bulan_2', 'bulan_3', 'bulan_4', 'bulan_5',
    'bulan_6', 'bulan_7', 'bulan_8', 'bulan_9', 'bulan_10', 'bulan_11',
    'bulan_12', 'occupation', 'financial_situation', 'delay_history',
    'communication_channel', 'working_time', 'payment_patterns',
    'payment_method'
]

# Custom transformer for home_ownership column
def encode_categorical_data(input_df):
    label_encoder = LabelEncoder()

    # Lakukan Label Encoding untuk kolom kategori yang sesuai
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

# Custom transformer untuk kolom home_ownership
class HomeOwnershipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[["home_ownership"]])
        return self

    def transform(self, X, y=None):
        X["home_ownership"] = X["home_ownership"].astype(int)
        X[["home_ownership"]] = self.scaler.fit_transform(X[["home_ownership"]])
        return X

# Custom transformer untuk kolom employment_status
class EmploymentStatusTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[["employment_status"]])
        return self

    def transform(self, X, y=None):
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
        X["employment_status"] = X["employment_status"].replace(employment_status_dict)
        X[["employment_status"]] = self.scaler.fit_transform(X[["employment_status"]])
        return X

def determine_score_category(collection_difficulty_score):
    if collection_difficulty_score >= 0 and collection_difficulty_score <= 281:
        collection_difficulty_category = "mudah"
    elif collection_difficulty_score >= 282 and collection_difficulty_score <= 372:
        collection_difficulty_category = "sedang"
    elif collection_difficulty_score >= 373 and collection_difficulty_score <= 1000:
        collection_difficulty_category = "sulit"
    else:
        collection_difficulty_category = "Skor Melebihi Parameter"
    return collection_difficulty_category

# Inisialisasi pipeline data
data_pipeline = Pipeline([
    ('home_ownership_transformer', HomeOwnershipTransformer()),
    ('employment_status_transformer', EmploymentStatusTransformer()),
])

def process_input_data(input_data):
    # Serialize the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode kolom kategori dengan label encoding
    input_df = encode_categorical_data(input_df)

    # Drop kolom yang tidak diperlukan
    input_df = input_df.drop(columns=unwanted_fields)

    # Gunakan pipeline data
    processed_data = data_pipeline.transform(input_df)

    # Convert hasil proses menjadi numpy array
    input_array = processed_data.values

    return input_array

def process_and_predict(input_data):
    # Proses input data
    input_array = process_input_data(input_data)

    # Prediksi dan kategori
    score, category = predict_collection_difficulty(input_array)

    return input_array, score, category

def predict_collection_difficulty(input_array):
    # Perform prediction using your model
    collection_difficulty_score = model.predict(input_array)[0]

    # Determine the prediction result based on criteria
    collection_difficulty_category = determine_score_category(collection_difficulty_score)

    return collection_difficulty_score, collection_difficulty_category