# This module contains every method for inference, as well as data transformation and preprocessing.

from ..apps import Fitur9Config
from ..libraries import utils

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib
"""
    This class calculates the loan income expenses ratio based on monthly payment, monthly income, and monthly expenses.
"""
class LoanIncomeExpensesRatioCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['loan_income_expenses_ratio'] = round(X['monthly_payment'] / (X['monthly_income'] - X['monthly_expenses']) * 100, 2)
        return X

"""
    This class calculates the default risk based on the loan income expenses ratio, asset value, and loan amount.
"""
class DefaultRiskCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['default_risk'] = X.apply(self.calculate_default_risk, axis=1)
        return X

    def calculate_default_risk(self, row):
        if row['loan_income_expenses_ratio'] < 20:
            return "Sangat Baik"
        elif 20 <= row['loan_income_expenses_ratio'] < 40 and row['asset_value'] * 0.2 > row['loan_amount']:
            return "Sangat Baik"
        elif 20 <= row['loan_income_expenses_ratio'] < 40 and row['asset_value'] * 0.5 > row['loan_amount']:
            return "Baik"
        elif 40 <= row['loan_income_expenses_ratio'] < 60 and row['asset_value'] * 0.2 >= row['loan_amount']:
            return "Baik"
        elif 20 <= row['loan_income_expenses_ratio'] < 40 and row['asset_value'] * 0.5 <= row['loan_amount']:
            return "Netral"
        elif 40 <= row['loan_income_expenses_ratio'] < 60 and row['asset_value'] * 0.5 >= row['loan_amount']:
            return "Beresiko"
        elif 40 <= row['loan_income_expenses_ratio'] < 60 and row['asset_value'] * 0.5 <= row['loan_amount']:
            return "Beresiko"
        elif 60 <= row['loan_income_expenses_ratio'] <= 80 and row['asset_value'] * 0.2 >= row['loan_amount']:
            return "Beresiko"
        elif 60 <= row['loan_income_expenses_ratio'] < 80 and row['asset_value'] * 0.5 <= row['loan_amount']:
            return "Sangat Beresiko"
        elif row['loan_income_expenses_ratio'] >= 80:
            return "Sangat Beresiko"
        elif row['loan_income_expenses_ratio'] >= 60:
            return "Beresiko"
        else:
            return None

class SESCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['ses'] = X.apply(self.calculate_ses, axis=1)
        return X

    def calculate_ses(self, row):
        if row['debtor_education_level'] == "SMA":
            if row['monthly_income'] < 3000000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 8000000 and row['asset_value'] < 400000000 and row['monthly_expenses'] <= 0.6 * row['monthly_income']:
                return "Rendah"
            elif row['monthly_income'] < 15000000 and row['asset_value'] < 800000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 30000000 and row['asset_value'] < 1000000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        elif row['debtor_education_level'] == "D3" or row['debtor_education_level'] == "D4":
            if row['monthly_income'] < 3500000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 10000000 and row['asset_value'] < 300000000 and row['monthly_expenses'] <= 0.6 * row['monthly_income']:
                return "Rendah"
            elif row['monthly_income'] < 20000000 and row['asset_value'] < 600000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 40000000 and row['asset_value'] < 800000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        elif row['debtor_education_level'] == "S1":
            if row['monthly_income'] < 5000000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 13000000 and row['asset_value'] < 300000000 and row['monthly_expenses']:
                return 'Rendah'
            elif row['monthly_income'] < 26000000 and row['asset_value'] < 600000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 52000000 and row['asset_value'] < 1000000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        elif row['debtor_education_level'] == "S2":
            if row['monthly_income'] < 7000000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 15000000 and row['asset_value'] < 400000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Rendah"
            elif row['monthly_income'] < 30000000 and row['asset_value'] < 800000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 60000000 and row['asset_value'] < 1200000000 and row['monthly_expenses'] <= 0.9 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        elif row['debtor_education_level'] == "S3":
            if row['monthly_income'] < 9000000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 18000000 and row['asset_value'] < 600000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Rendah"
            elif row['monthly_income'] < 36000000 and row['asset_value'] < 1000000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 72000000 and row['asset_value'] < 1500000000 and row['monthly_expenses'] <= 0.9 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        else:
            return "Tidak Diketahui"  # Tingkat pendidikan tidak dikenali
    
"""
    This class performs categorical encoding for various features such as debtor education level, marital status, SES, total capital, default risk, loan purpose, and default potential.
"""
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = {
            "debtor_education_level": {
                "SD": 0, "SMP": 1, "SMA": 2, "D3": 3, "D4": 4, "S1": 5, "S2": 6, "S3": 7
            },
            "debtor_marital_status": {
                "Belum menikah": 0, "Sudah menikah": 1, "Cerai Hidup": 2, "Cerai Mati": 3
            },
            "ses": {
                "Sangat Rendah": 0, "Rendah": 1, "Menengah": 2, "Tinggi": 3, "Sangat Tinggi": 4
            },
            "clasify_total_capital": {
                "Sangat Kuat": 1, "Cukup": 2, "Lemah": 3, "Kuat": 4, "Sangat Lemah": 0
            },
            "default_risk": {
                "Baik": 1, "Sangat Baik": 0, "Netral": 2, "Beresiko": 3, "Sangat Beresiko": 4
            },
            "loan_purpose": {
                "kredit usaha": 1, "kredit kendaraan bermotor": 2, "kredit pribadi": 3, "pernikahan": 4, "lainnya": 0, "kredit properti": 1, "kendaraan bermotor": 0
            },
            "default_potential" : {"Sangat Baik" : 0,  "Baik" : 1, "Netral" : 2, "Buruk" : 3, "Sangat Buruk" : 4, "Suspicious" : 5}
        }

    def fit(self, X, y=None):
        return self


data_transformation_pipeline = Pipeline([
    # ('loan_income_expenses_ratio', LoanIncomeExpensesRatioCalculator()),
    # ('default_risk_calculator', DefaultRiskCalculator()),
    # ('ses_calculator', SESCalculator()),
])


categorical_preprocessing = Pipeline([
    ('categorical_encoder', CategoricalEncoder()),
])
"""
    This function predicts the recommended solution based on the input data using a pre-trained model and scaler.
"""


def predict_recommended_solution(data):
    
    default_solution_model = Fitur9Config.default_solution_model
    default_solution_scaler = Fitur9Config.default_solution_scaler
    
    default_kredit_dataset = "data_user_recomedation_v3.csv"
    
    df = transform_input(data)
    
    solution_df  = df[[ 'monthly_income', 'loan_purpose']]
    solution_df = default_solution_scaler.transform(solution_df)
    df['recommendation'] = default_solution_model.predict(solution_df)
    
    utils.append_new_row(default_kredit_dataset, df)
    
    data = {
        "recommendation": df['recommendation'].values[0],
    }
    
    return data
    
"""
    This function transforms the input data into a pandas DataFrame for further processing.
"""    
    
def transform_input(data):
  data = {key: [value] for key, value in data.items()}
  df = pd.DataFrame(data)
  
  return df
