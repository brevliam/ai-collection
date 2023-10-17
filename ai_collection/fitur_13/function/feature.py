# This module contains every method for inference, as well as data transformation and preprocessing.
from ..apps import Fitur13Config
from ..libraries import utils
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
import joblib
import pandas as pd

def predict_best_time_to_remind(data):
    model = Fitur13Config.best_time_to_remind_model
    DATASET_FILE_NAME = 'Reminder_AI_Reschedule_Automation_V04_20231004.xlsx'
    
    input_df = transform_input_debtor(data)
    preprocessed_df = data_preprocessing_best_time_reminder.fit_transform(input_df)
    output = model.predict(preprocessed_df)
    output_proba = model.predict_proba(preprocessed_df).flatten()
    
    result = transform_best_time_to_remind_pred_output(output, output_proba)
    
    utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
    
    return result

def predict_best_time_to_follow_up(data):
    model = Fitur13Config.best_time_to_follow_up_model
    DATASET_FILE_NAME = 'Follow_Up_AI_Reschedule_Automation_V07_20231004.xlsx'
    
    input_df = transform_input_debtor(data)
    preprocessed_df = data_preprocessing_best_time_follow_up.fit_transform(input_df)
    # Drop instances where debtor_aging is 1
    preprocessed_df = preprocessed_df[preprocessed_df['debtor_aging'] != 1]
    output = model.predict(preprocessed_df)
    output_proba = model.predict_proba(preprocessed_df).flatten()
    
    result = transform_best_time_to_follow_up_pred_output(output, output_proba)
    
    utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
    
    return result

def predict_reschedule(data):
    model = Fitur13Config.reschedule_model
    DATASET_FILE_NAME = 'AI_Reschedule_data_terbaru.xlsx'
    
    input_df = transform_input_debtor(data)
    # Drop instances where debtor_aging is 1

    output = model.predict(input_df)
    # output_proba = model.predict_proba(input_df).flatten()
    
    result = transform_reschedule(output)
    
    utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
    
    return result

def transform_input_debtor(data):
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
  
    return df

def transform_reschedule(output):
    label_mapping = {0: 'Layak Reschedule Dengan Resiko Rendah',
                     1: 'Layak Reschedule Dengan Resiko Sedang',
                     2: 'Tidak Layak Reschedule Dengan Resiko Sedang',
                     3: 'Tidak Layak Reschedule Dengan Resiko Tinggi'}
    reschedule_eligibility = label_mapping[output[0]]
    result = {
        'reschedule_eligibility': reschedule_eligibility
    }

    return result

def transform_best_time_to_remind_pred_output(output, output_proba):
    label_mapping = {0: 'Pagi', 1: 'Siang', 2: 'Sore', 3: 'Malam'}
    best_time_to_remind = label_mapping[output[0]]
    best_time_to_remind_proba = rank_best_time_to_remind_output(output_proba, label_mapping)
    result = {
        'best_time_to_remind': best_time_to_remind,
        'best_time_to_remind_probability': best_time_to_remind_proba
    }

    return result

def transform_best_time_to_follow_up_pred_output(output, output_proba):
    label_mapping = {0: 'Pagi', 1: 'Siang', 2: 'Sore', 3: 'Malam'}
    best_time_to_follow_up = label_mapping[output[0]]
    best_time_to_follow_up_proba = rank_best_time_to_follow_up_output(output_proba, label_mapping)
    result = {
        'best_time_to_follow_up': best_time_to_follow_up,
        'best_time_to_follow_up_probability': best_time_to_follow_up_proba
    }

    return result

def rank_best_time_to_remind_output(result_proba, label_mapping):
    sorted_proba_indices = (-result_proba).argsort()
    rank_list = []
    for rank, idx in enumerate(sorted_proba_indices):
        label = label_mapping[idx]
        rank_list.append(f"{label} : {result_proba[idx]:.2%}")
    return rank_list  

def rank_best_time_to_follow_up_output(result_proba, label_mapping):
    sorted_proba_indices = (-result_proba).argsort()
    rank_list = []
    for rank, idx in enumerate(sorted_proba_indices):
        label = label_mapping[idx]
        rank_list.append(f"{label} : {result_proba[idx]:.2%}")
    return rank_list  

def rank_reschedule_eligibility_output(result_proba, label_mapping):
    sorted_proba_indices = (-result_proba).argsort()
    rank_list = []
    for rank, idx in enumerate(sorted_proba_indices):
        label = label_mapping[idx]
        rank_list.append(f"{label} : {result_proba[idx]:.2%}")
    return rank_list  

class ObjectToCategoriesReminder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X["debtor_working_time"] = X["debtor_working_time"].astype("category")
        X["debtor_previous_communication_channel"] = X["debtor_previous_communication_channel"].astype("category")
        X["last_interaction_type"] = X["last_interaction_type"].astype("category")
        X["reminder_response"] = X["reminder_response"].astype("category")
        return X
    
class ObjectToCategoriesFollowUp(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X["debtor_aging"] = X["debtor_aging"].astype("category")
        X["debtor_working_time"] = X["debtor_working_time"].astype("category")
        X["debtor_previous_communication_channel"] = X["debtor_previous_communication_channel"].astype("category")
        X["debtor_field_communication"] = X["debtor_field_communication"].astype("category")
        X["last_interaction_type"] = X["last_interaction_type"].astype("category")
        X["follow_up_response"] = X["follow_up_response"].astype("category")
        return X

class DebtorWorkingTimeReminderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df_working_time = X[['debtor_working_time']]
        df_working_time = self.transform_debt_working_time(df_working_time)
        # Check if the columns already exist
        if 'busy_pagi' not in df_working_time.columns:
            df_working_time['busy_pagi'] = df_working_time.apply(lambda row: self._check_busy_pagi(row), axis=1)
        if 'busy_siang' not in df_working_time.columns:
            df_working_time['busy_siang'] = df_working_time.apply(lambda row: self._check_busy_siang(row), axis=1)
        if 'busy_sore' not in df_working_time.columns:
            df_working_time['busy_sore'] = df_working_time.apply(lambda row: self._check_busy_sore(row), axis=1)
        if 'busy_malam' not in df_working_time.columns:
            df_working_time['busy_malam'] = df_working_time.apply(lambda row: self._check_busy_malam(row), axis=1)
        return df_working_time
    
    def transform_debt_working_time(self, df_working_time):
        debtor_working_time_dummy = []
        if df_working_time.values[0] == 'Malam-Pagi':
          debtor_working_time_dummy.append(1)
        else:
          debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Pagi-Malam':
          debtor_working_time_dummy.append(1)
        else:
          debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Pagi-Siang':
          debtor_working_time_dummy.append(1)
        else:
          debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Pagi-Sore':
          debtor_working_time_dummy.append(1)
        else:
          debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Siang-Malam':
          debtor_working_time_dummy.append(1)
        else:
          debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Siang-Sore':
          debtor_working_time_dummy.append(1)
        else:
          debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Siang-Sore':
          debtor_working_time_dummy.append(1)
        else:
          debtor_working_time_dummy.append(0)

        debtor_work_dummy = np.array(debtor_working_time_dummy).reshape((1, 7))

        debtor_work_df = pd.DataFrame(debtor_work_dummy, columns=['debtor_working_time_Malam-Pagi', 
                              'debtor_working_time_Pagi-Malam', 
                              'debtor_working_time_Pagi-Siang', 
                              'debtor_working_time_Pagi-Sore', 
                              'debtor_working_time_Siang-Malam',
                              'debtor_working_time_Siang-Sore',
                              'debtor_working_time_Sore-Malam'])  
        
        return debtor_work_df
    
    def _check_busy_pagi(self, row):
        return 1 if (row['debtor_working_time_Pagi-Siang'] == 1) or (row['debtor_working_time_Pagi-Sore'] == 1) or (row['debtor_working_time_Pagi-Malam'] == 1) else 0

    def _check_busy_siang(self, row):
        return 1 if (row['debtor_working_time_Pagi-Siang'] == 1) or (row['debtor_working_time_Pagi-Sore'] == 1) or (row['debtor_working_time_Pagi-Malam'] == 1) or (row['debtor_working_time_Siang-Malam'] == 1) else 0

    def _check_busy_sore(self, row):
        return 1 if (row['debtor_working_time_Pagi-Malam'] == 1) or (row['debtor_working_time_Pagi-Sore'] == 1) or (row['debtor_working_time_Siang-Malam'] == 1) else 0

    def _check_busy_malam(self, row):
        return 1 if (row['debtor_working_time_Pagi-Malam'] == 1) or (row['debtor_working_time_Siang-Malam'] == 1) or (row['debtor_working_time_Malam-Pagi'] == 1) else 0
   
    
class DebtorWorkingTimeFollowUpTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df_aging = X[['debtor_aging']]
        df_working_time = X[['debtor_working_time']]
        df_working_time = self.transform_debt_working_time(df_working_time)
        df = pd.concat([df_aging, df_working_time], axis = 1)
        # Check if the columns already exist
        if 'busy_pagi' not in df.columns:
            df['busy_pagi'] = df.apply(lambda row: self._check_busy_pagi(row), axis=1)
        if 'busy_siang' not in df.columns:
            df['busy_siang'] = df.apply(lambda row: self._check_busy_siang(row), axis=1)
        if 'busy_sore' not in df.columns:
            df['busy_sore'] = df.apply(lambda row: self._check_busy_sore(row), axis=1)
        if 'busy_malam' not in df.columns:
            df['busy_malam'] = df.apply(lambda row: self._check_busy_malam(row), axis=1)
        return df

    def transform_debt_working_time(self, df_working_time):
        debtor_working_time_dummy = []

        if df_working_time.iloc[0].iloc[0] == 'Malam-Pagi':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)

        if df_working_time.iloc[0].iloc[0] == 'Pagi-Malam':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)

        if df_working_time.iloc[0].iloc[0] == 'Pagi-Siang':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)

        if df_working_time.iloc[0].iloc[0] == 'Pagi-Sore':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)

        if df_working_time.iloc[0].iloc[0] == 'Siang-Malam':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)

        if df_working_time.iloc[0].iloc[0] == 'Siang-Sore':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)

        if df_working_time.iloc[0].iloc[0] == 'Sore-Malam':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)

        debtor_work_dummy = np.array(debtor_working_time_dummy).reshape((1, 7))

        debtor_work_df = pd.DataFrame(debtor_work_dummy, columns=['debtor_working_time_Malam-Pagi',
                                                                  'debtor_working_time_Pagi-Malam',
                                                                  'debtor_working_time_Pagi-Siang',
                                                                  'debtor_working_time_Pagi-Sore',
                                                                  'debtor_working_time_Siang-Malam',
                                                                  'debtor_working_time_Siang-Sore',
                                                                  'debtor_working_time_Sore-Malam'])

        return debtor_work_df


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
        X['best_time_to_remind'] = X.apply(self.find_best_time, axis=1)
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
            "best_time_to_remind": {"Pagi" : 0, 
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
    
data_preprocessing_best_time_reminder = Pipeline([
    ('object_to_categories', ObjectToCategoriesReminder()),
    ('debtor_working_time_reminder_transformer', DebtorWorkingTimeReminderTransformer()),
])

data_preprocessing_best_time_follow_up = Pipeline([
    ('object_to_categories', ObjectToCategoriesFollowUp()),
    ('debtor_working_time_follow_up_transformer', DebtorWorkingTimeFollowUpTransformer()),
])

