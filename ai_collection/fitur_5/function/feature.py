# This module contains every method for inference, as well as data transformation and preprocessing.
from ..apps import Fitur5Config
from ..libraries import utils
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
import joblib
import pandas as pd

def predict_best_time_to_bill(data):
    model = Fitur5Config.besttime_to_bill_model
    DATASET_FILE_NAME = 'debtor_v05_231005.csv'
    
    input_df = transform_input_debtor(data)
    preprocessed_df = data_preprocessing_best_time.fit_transform(input_df)
    output = model.predict(preprocessed_df)
    output_proba = model.predict_proba(preprocessed_df).flatten()
    
    result = transform_best_time_to_bill_pred_output(output, output_proba)
    
    utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
    
    return result

def transform_input_debtor(data):
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
  
    return df


def transform_best_time_to_bill_pred_output(output, output_proba):
    label_mapping = {0: 'Pagi', 1: 'Siang', 2: 'Sore', 3: 'Malam'}
    best_time_to_bill = label_mapping[output[0]]
    best_time_to_bill_proba = rank_best_time_to_bill_output(output_proba, label_mapping)
    result = {
        'best_time_to_bill': best_time_to_bill,
        'best_time_to_bill_probability': best_time_to_bill_proba
    }

    return result

def rank_best_time_to_bill_output(result_proba, label_mapping):
    sorted_proba_indices = (-result_proba).argsort()
    rank_list = []
    for rank, idx in enumerate(sorted_proba_indices):
        label = label_mapping[idx]
        rank_list.append(f"{label} : {result_proba[idx]:.2%}")
    return rank_list  


class ObjectToCategories(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X["debtor_gender"] = X["debtor_gender"].astype("category")
        X["debtor_education_level"] = X["debtor_education_level"].astype("category")
        X["employment_status"] = X["employment_status"].astype("category")
        X["debtor_working_day"] = X["debtor_working_day"].astype("category")
        X["debtor_working_time"] = X["debtor_working_time"].astype("category")
        return X

class DebtorWorkingTimeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        debt_working_time = X[['debtor_working_time']]
        debt_working_time = self.transform_debt_working_time(debt_working_time)
        # Check if the columns already exist
        if 'busy_pagi' not in debt_working_time.columns:
            debt_working_time['busy_pagi'] = debt_working_time.apply(lambda row: self._check_busy_pagi(row), axis=1)
        if 'busy_siang' not in debt_working_time.columns:
            debt_working_time['busy_siang'] = debt_working_time.apply(lambda row: self._check_busy_siang(row), axis=1)
        if 'busy_sore' not in debt_working_time.columns:
            debt_working_time['busy_sore'] = debt_working_time.apply(lambda row: self._check_busy_sore(row), axis=1)
        if 'busy_malam' not in debt_working_time.columns:
            debt_working_time['busy_malam'] = debt_working_time.apply(lambda row: self._check_busy_malam(row), axis=1)
        return debt_working_time
    
    def transform_debt_working_time(self, debt_working_time):
        debt_working_time_dum = []
        if debt_working_time.values[0] == 'Malam-Pagi':
          debt_working_time_dum.append(1)
        else:
          debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Pagi-Malam':
          debt_working_time_dum.append(1)
        else:
          debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Pagi-Siang':
          debt_working_time_dum.append(1)
        else:
          debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Pagi-Sore':
          debt_working_time_dum.append(1)
        else:
          debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Siang-Malam':
          debt_working_time_dum.append(1)
        else:
          debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Siang-Sore':
          debt_working_time_dum.append(1)
        else:
          debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Sore-Malam':
          debt_working_time_dum.append(1)
        else:
          debt_working_time_dum.append(0)

        debt_work_dum = np.array(debt_working_time_dum).reshape((1, 7))

        debt_work_df = pd.DataFrame(debt_work_dum, columns=['debtor_working_time_Malam-Pagi', 
                              'debtor_working_time_Pagi-Malam', 
                              'debtor_working_time_Pagi-Siang', 
                              'debtor_working_time_Pagi-Sore', 
                              'debtor_working_time_Siang-Malam',
                              'debtor_working_time_Siang-Sore',
                              'debtor_working_time_Sore-Malam'])  
        
        return debt_work_df

    def _check_busy_pagi(self, row):
        return 1 if (row['debtor_working_time_Pagi-Siang'] == 1) or (row['debtor_working_time_Pagi-Sore'] == 1) or (row['debtor_working_time_Pagi-Malam'] == 1) else 0

    def _check_busy_siang(self, row):
        return 1 if (row['debtor_working_time_Pagi-Siang'] == 1) or (row['debtor_working_time_Pagi-Sore'] == 1) or (row['debtor_working_time_Pagi-Malam'] == 1) or (row['debtor_working_time_Siang-Malam'] == 1) else 0

    def _check_busy_sore(self, row):
        return 1 if (row['debtor_working_time_Pagi-Malam'] == 1) or (row['debtor_working_time_Pagi-Sore'] == 1) or (row['debtor_working_time_Siang-Malam'] == 1) else 0

    def _check_busy_malam(self, row):
        return 1 if (row['debtor_working_time_Pagi-Malam'] == 1) or (row['debtor_working_time_Siang-Malam'] == 1) or (row['debtor_working_time_Malam-Pagi'] == 1) else 0
    

class FindBestTimeTarget(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['best_time_to_bill'] = X.apply(self.find_best_time, axis=1)
        return X

    def find_best_time(row):
        if (row['busy_pagi'] == 0) and (row['busy_siang'] == 0) and (row['busy_sore'] == 0) and (row['busy_malam'] == 0):
            return np.random.choice(["Pagi", "Siang", "Sore", "Malam"], p=[0.4, 0.4, 0.15, 0.05])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 0) and (row['busy_sore'] == 0) and (row['busy_malam'] == 1):
            return np.random.choice(["Pagi", "Siang", "Sore"], p=[0.4, 0.4, 0.2])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 0) and (row['busy_sore'] == 1) and (row['busy_malam'] == 0):
            return np.random.choice(["Pagi", "Siang", "Malam"], p=[0.4, 0.5, 0.1])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 1) and (row['busy_sore'] == 0) and (row['busy_malam'] == 0):
            return np.random.choice(["Pagi", "Sore", "Malam"], p=[0.5, 0.4, 0.1])
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 0) and (row['busy_sore'] == 0) and (row['busy_malam'] == 0):
            return np.random.choice(["Siang", "Sore", "Malam"], p=[0.6, 0.3, 0.1])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 0) and (row['busy_sore'] == 1) and (row['busy_malam'] == 1):
            return np.random.choice(["Pagi", "Siang"], p=[0.4, 0.6])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 1) and (row['busy_sore'] == 0) and (row['busy_malam'] == 1):
            return np.random.choice(["Pagi", "Sore"], p=[0.55, 0.45])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 1) and (row['busy_sore'] == 1) and (row['busy_malam'] == 0):
            return np.random.choice(["Pagi", "Malam"], p=[0.7, 0.3])
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 1) and (row['busy_sore'] == 0) and (row['busy_malam'] == 0):
            return np.random.choice(["Sore", "Malam"], p=[0.55, 0.45])
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 0) and (row['busy_sore'] == 1) and (row['busy_malam'] == 0):
            return np.random.choice(["Siang", "Malam"], p=[0.8, 0.2])
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 0) and (row['busy_sore'] == 0) and (row['busy_malam'] == 1):
            return np.random.choice(["Siang", "Sore"], p=[0.8, 0.2])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 1) and (row['busy_sore'] == 1) and (row['busy_malam'] == 1):
            return "Pagi"
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 0) and (row['busy_sore'] == 1) and (row['busy_malam'] == 1):
            return "Siang"
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 1) and (row['busy_sore'] == 0) and (row['busy_malam'] == 1):
            return "Sore"
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 1) and (row['busy_sore'] == 1) and (row['busy_malam'] == 0):
            return "Malam"
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 1) and (row['busy_sore'] == 1) and (row['busy_malam'] == 1):
            return np.random.choice(["Siang", "Malam"], p=[0.6, 0.4])
        else:
            return None
        

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = {
            "best_time_to_bill": {"Pagi" : 0, 
                                  "Siang": 1,
                                  "Sore" : 2,
                                  "Malam": 3},
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for column, mapping in self.mapping.items():
            if column in X_encoded.columns and X_encoded[column].dtype == 'object':
                X_encoded[column] = X_encoded[column].map(mapping)
        return X_encoded
    
data_preprocessing_best_time = Pipeline([
    ('object_to_categories', ObjectToCategories()),
    ('debt_working_time_transformer', DebtorWorkingTimeTransformer()),
])

