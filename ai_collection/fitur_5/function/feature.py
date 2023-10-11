# This module contains every method for inference, as well as data transformation and preprocessing.
from ..apps import Fitur5Config
from ..libraries import utils
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA
import math
import pandas as pd
import numpy as np
from datetime import datetime

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

def predict_recommended_collectors_assignments(data):
    model = Fitur5Config.recsys_collector_assignments_model
    DATASET_FILE_NAME_COLL = 'collector_v03_231005.csv'

    input_debtor_df = transform_input_debtor(data)
    dataset_coll_path = utils.load_dataset_path(DATASET_FILE_NAME_COLL)
    dt_collector = pd.read_csv(dataset_coll_path)

    preprocessed_debtor = data_preprocessing_debtor.fit_transform(input_debtor_df)
    preprocessed_collector = data_preprocessing_collector.fit_transform(dt_collector)
    xdebt, xcoll = scale_for_recsys(preprocessed_debtor, preprocessed_collector)

    output = model.predict([xdebt, xcoll])
    result = transform_recommended_collectors_output(output, input_debtor_df, dt_collector)
    
    return result

def predict_interaction_efficiency(data):
    model = Fitur5Config.interaction_eficiency_model
    DATASET_FILE_NAME_INTER = 'collector_performances_v03_231005.csv'
 
    input_df = transform_input_debtor(data)
    dataset_inter_path = utils.load_dataset_path(DATASET_FILE_NAME_INTER)
    interactions_df = pd.read_csv(dataset_inter_path)

    preprocessed_input = data_preprocessing_input_inter.fit_transform(input_df).reset_index(drop=True)
    preprocessed_inter = data_preprocessing_interactions.fit_transform(interactions_df).reset_index(drop=True)
    preprocessed_input = preprocessed_input.reset_index(drop=True)
    preprocessed_inter = preprocessed_inter.reset_index(drop=True)
    combined_df = pd.concat([preprocessed_inter, preprocessed_input], ignore_index=True)
    pca_df = scale_and_pca.fit_transform(combined_df)

    y_clust = model.predict(pca_df)
    combined_df["clusters"] = y_clust
    
    result = transform_interaction_eficiency_output(combined_df)
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

def transform_recommended_collectors_output(output, input_debtor_df, dt_collector):
    output = pd.DataFrame(output)
    output["distance"] = calculate_distance_per_collector(input_debtor_df, dt_collector)
    output.sort_values(by=[0, 'distance'], ascending=[False, True], inplace=True)
    recommended_collectors_index = output[:10].index
    recommended_collectors = dt_collector.loc[recommended_collectors_index, 'collector_name'].tolist()
    result = {
        'recommended_collectors_to_assign': recommended_collectors
    }

    return result

def transform_interaction_eficiency_output(combined_df):
    # takes the output clusters on input data
    cluster = combined_df['clusters'].iloc[-1]
    if cluster == 0:
       cat_cluster = "Efisien dalam Interaksi dengan Pelanggan"
    elif cluster == 1:
       cat_cluster = "Efisien dalam Mobilitas/Perjalanan"
    elif cluster == 2:
       cat_cluster = "Efisien dalam Manajemen Waktu"
    elif cluster == 3:
       cat_cluster = "Efisien dalam Proses Aktifitas"
    elif cluster == 4:
       cat_cluster = "Efisien dalam Respon dan Interaksi"
    result = {
        'category_cluster': cat_cluster
    }

    return result

def rank_best_time_to_bill_output(result_proba, label_mapping):
    sorted_proba_indices = (-result_proba).argsort()
    rank_list = []
    for rank, idx in enumerate(sorted_proba_indices):
        label = label_mapping[idx]
        rank_list.append(f"{label} : {result_proba[idx]:.2%}")
    return rank_list  


class ObjectToCategoriesDebtor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categorical_columns = ['debtor_gender', 'debtor_education_level',
                               'employment_status', 'debtor_working_day',
                               'debtor_working_time']

        X[categorical_columns] = X[categorical_columns].astype('category')

        return X

class ObjectToCategoriesCollector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categorical_columns = ['collector_education_level', 'collector_workday', 'collector_worktime']

        X[categorical_columns] = X[categorical_columns].astype('category')

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
            "transportation_type": {"Motor" : 0,
                                    "Mobil": 1}
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for column, mapping in self.mapping.items():
            if column in X_encoded.columns and X_encoded[column].dtype == 'object':
                X_encoded[column] = X_encoded[column].map(mapping)
        return X_encoded




class PreprocessDebtorForRecSys(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt_debt = X[['debtor_age', 'debtor_latitude',
                       'debtor_longitude']]

        debt_edulvl = self._transform_debt_edu_level(X['debtor_education_level'])
        debt_workday = self._transform_debt_working_day(X['debtor_working_day'])
        debt_worktime = self._transform_debt_working_time(X['debtor_working_time'])
        dt_debtor = pd.concat([dt_debt, debt_edulvl, debt_workday, debt_worktime], axis=1)
        dt_debtor = self._generate_luang_debtor(dt_debtor)
        dt_debtor = pd.concat([dt_debtor] * 1000)

        return dt_debtor.reset_index(drop=True)

    def _transform_debt_edu_level(self, debt_edu_level):
        debt_edu_level_dum = []
        if debt_edu_level.values[0] == 'D3':
          debt_edu_level_dum.append(1)
        else:
          debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'D4':
          debt_edu_level_dum.append(1)
        else:
          debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'S1':
          debt_edu_level_dum.append(1)
        else:
          debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'S2':
          debt_edu_level_dum.append(1)
        else:
          debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'S3':
          debt_edu_level_dum.append(1)
        else:
          debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'SD':
          debt_edu_level_dum.append(1)
        else:
          debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'SMA':
          debt_edu_level_dum.append(1)
        else:
          debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'SMP':
          debt_edu_level_dum.append(1)
        else:
          debt_edu_level_dum.append(0)


        debt_edu_dum = np.array(debt_edu_level_dum).reshape((1, 8))

        debt_edu_df = pd.DataFrame(debt_edu_dum, columns=['debtor_education_level_D3','debtor_education_level_D4',
                                                            'debtor_education_level_S1','debtor_education_level_S2',
                                                            'debtor_education_level_S3','debtor_education_level_SD',
                                                            'debtor_education_level_SMA','debtor_education_level_SMP'])

        return debt_edu_df

    def _transform_debt_working_day(self, debt_working_day):
        debt_working_day_dum = []
        if debt_working_day.values[0] == 'Sabtu-Minggu':
          debt_working_day_dum.append(1)
        else:
          debt_working_day_dum.append(0)
        if debt_working_day.values[0] == 'Senin-Jumat':
          debt_working_day_dum.append(1)
        else:
          debt_working_day_dum.append(0)
        if debt_working_day.values[0] == 'Senin-Minggu':
          debt_working_day_dum.append(1)
        else:
          debt_working_day_dum.append(0)


        debt_work_dum = np.array(debt_working_day_dum).reshape((1, 3))

        debt_work_df = pd.DataFrame(debt_work_dum, columns=['debtor_workday_Sabtu-Minggu',
                              'debtor_workday_Senin-Jumat',
                              'debtor_workday_Senin-Minggu'])

        return debt_work_df

    def _transform_debt_working_time(self, debt_working_time):
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

        debt_work_df = pd.DataFrame(debt_work_dum, columns=['debtor_worktime_Malam-Pagi',
                              'debtor_worktime_Pagi-Malam',
                              'debtor_worktime_Pagi-Siang',
                              'debtor_worktime_Pagi-Sore',
                              'debtor_worktime_Siang-Malam',
                              'debtor_worktime_Siang-Sore',
                              'debtor_worktime_Sore-Malam'])

        return debt_work_df

    def _generate_luang_debtor(self, dt_debtor):
        # Check if the columns already exist
        if 'debtor_luang_pagi' not in dt_debtor.columns:
            dt_debtor['debtor_luang_pagi'] = dt_debtor.apply(lambda row: self._check_debtor_luang_pagi(row), axis=1)
        if 'debtor_luang_siang' not in dt_debtor.columns:
            dt_debtor['debtor_luang_siang'] = dt_debtor.apply(lambda row: self._check_debtor_luang_siang(row), axis=1)
        if 'debtor_luang_sore' not in dt_debtor.columns:
            dt_debtor['debtor_luang_sore'] = dt_debtor.apply(lambda row: self._check_debtor_luang_sore(row), axis=1)
        if 'debtor_luang_malam' not in dt_debtor.columns:
            dt_debtor['debtor_luang_malam'] = dt_debtor.apply(lambda row: self._check_debtor_luang_malam(row), axis=1)
        if 'debtor_luang_senin' not in dt_debtor.columns:
            dt_debtor['debtor_luang_senin'] = dt_debtor.apply(lambda row: self._check_debtor_luang_senin(row), axis=1)
        if 'debtor_luang_selasa' not in dt_debtor.columns:
            dt_debtor['debtor_luang_selasa'] = dt_debtor.apply(lambda row: self._check_debtor_luang_selasa(row), axis=1)
        if 'debtor_luang_rabu' not in dt_debtor.columns:
            dt_debtor['debtor_luang_rabu'] = dt_debtor.apply(lambda row: self._check_debtor_luang_rabu(row), axis=1)
        if 'debtor_luang_kamis' not in dt_debtor.columns:
            dt_debtor['debtor_luang_kamis'] = dt_debtor.apply(lambda row: self._check_debtor_luang_kamis(row), axis=1)
        if 'debtor_luang_jumat' not in dt_debtor.columns:
            dt_debtor['debtor_luang_jumat'] = dt_debtor.apply(lambda row: self._check_debtor_luang_jumat(row), axis=1)
        if 'debtor_luang_sabtu' not in dt_debtor.columns:
            dt_debtor['debtor_luang_sabtu'] = dt_debtor.apply(lambda row: self._check_debtor_luang_sabtu(row), axis=1)
        if 'debtor_luang_minggu' not in dt_debtor.columns:
            dt_debtor['debtor_luang_minggu'] = dt_debtor.apply(lambda row: self._check_debtor_luang_minggu(row), axis=1)

        dt_debtor.drop(['debtor_workday_Sabtu-Minggu','debtor_workday_Senin-Jumat',
                   'debtor_workday_Senin-Minggu', 'debtor_worktime_Malam-Pagi',
                   'debtor_worktime_Pagi-Malam', 'debtor_worktime_Pagi-Siang',
                   'debtor_worktime_Pagi-Sore', 'debtor_worktime_Siang-Malam',
                   'debtor_worktime_Siang-Sore', 'debtor_worktime_Sore-Malam'], axis=1, inplace=True)
        return dt_debtor

    def _check_debtor_luang_pagi(self, row):
        return 0 if (row['debtor_worktime_Pagi-Siang'] == 1) or (row['debtor_worktime_Pagi-Sore'] == 1) or (row['debtor_worktime_Pagi-Malam'] == 1)  else 1

    def _check_debtor_luang_siang(self, row):
        return 0 if (row['debtor_worktime_Pagi-Siang'] == 1) or (row['debtor_worktime_Pagi-Sore'] == 1) or (row['debtor_worktime_Pagi-Malam'] == 1) or (row['debtor_worktime_Siang-Malam'] == 1) or (row['debtor_worktime_Siang-Sore'] == 1) else 1

    def _check_debtor_luang_sore(self, row):
        return 0 if (row['debtor_worktime_Pagi-Malam'] == 1) or (row['debtor_worktime_Pagi-Sore'] == 1) or (row['debtor_worktime_Siang-Malam'] == 1) or (row['debtor_worktime_Siang-Sore'] == 1) or (row['debtor_worktime_Sore-Malam'] == 1)  else 1

    def _check_debtor_luang_malam(self, row):
        return 0 if (row['debtor_worktime_Pagi-Malam'] == 1) or (row['debtor_worktime_Siang-Malam'] == 1) or (row['debtor_worktime_Malam-Pagi'] == 1) or (row['debtor_worktime_Sore-Malam'] == 1)  else 1

    def _check_debtor_luang_senin(self, row):
        return 0 if (row['debtor_workday_Senin-Jumat'] == 1) or (row['debtor_workday_Senin-Minggu'] == 1) else 1

    def _check_debtor_luang_selasa(self, row):
        return 0 if (row['debtor_workday_Senin-Jumat'] == 1) or (row['debtor_workday_Senin-Minggu'] == 1) else 1

    def _check_debtor_luang_rabu(self, row):
        return 0 if (row['debtor_workday_Senin-Jumat'] == 1) or (row['debtor_workday_Senin-Minggu'] == 1) else 1

    def _check_debtor_luang_kamis(self, row):
        return 0 if (row['debtor_workday_Senin-Jumat'] == 1) or (row['debtor_workday_Senin-Minggu'] == 1) else 1

    def _check_debtor_luang_jumat(self, row):
        return 0 if (row['debtor_workday_Senin-Jumat'] == 1) or (row['debtor_workday_Senin-Minggu'] == 1) else 1

    def _check_debtor_luang_sabtu(self, row):
        return 0 if (row['debtor_workday_Sabtu-Minggu'] == 1) or (row['debtor_workday_Senin-Minggu'] == 1) else 1

    def _check_debtor_luang_minggu(self, row):
        return 0 if (row['debtor_workday_Sabtu-Minggu'] == 1) or (row['debtor_workday_Senin-Minggu'] == 1) else 1


class PreprocessCollectorForRecSys(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt_collector = X[['collector_age','collector_latitude', 'collector_longitude',
                          'collector_education_level','collector_workday','collector_worktime']]

        dt_collector = pd.get_dummies(dt_collector, columns=['collector_education_level',
                                                      'collector_workday','collector_worktime'])
        dt_collector = self._generate_kerja_collector(dt_collector)
        dt_collector = self._adjust_education_level(dt_collector)

        return dt_collector.reset_index(drop=True)

    def _generate_kerja_collector(self, dt_collector):
        # Check if the columns already exist
        if 'collector_kerja_pagi' not in dt_collector.columns:
            dt_collector['collector_kerja_pagi'] = dt_collector.apply(lambda row: self._check_collector_kerja_pagi(row), axis=1)
        if 'collector_kerja_siang' not in dt_collector.columns:
            dt_collector['collector_kerja_siang'] = dt_collector.apply(lambda row: self._check_collector_kerja_siang(row), axis=1)
        if 'collector_kerja_sore' not in dt_collector.columns:
            dt_collector['collector_kerja_sore'] = dt_collector.apply(lambda row: self._check_collector_kerja_sore(row), axis=1)
        if 'collector_kerja_malam' not in dt_collector.columns:
            dt_collector['collector_kerja_malam'] = dt_collector.apply(lambda row: self._check_collector_kerja_malam(row), axis=1)
        if 'collector_kerja_senin' not in dt_collector.columns:
            dt_collector['collector_kerja_senin'] = dt_collector.apply(lambda row: self._check_collector_kerja_senin(row), axis=1)
        if 'collector_kerja_selasa' not in dt_collector.columns:
            dt_collector['collector_kerja_selasa'] = dt_collector.apply(lambda row: self._check_collector_kerja_selasa(row), axis=1)
        if 'collector_kerja_rabu' not in dt_collector.columns:
            dt_collector['collector_kerja_rabu'] = dt_collector.apply(lambda row: self._check_collector_kerja_rabu(row), axis=1)
        if 'collector_kerja_kamis' not in dt_collector.columns:
            dt_collector['collector_kerja_kamis'] = dt_collector.apply(lambda row: self._check_collector_kerja_kamis(row), axis=1)
        if 'collector_kerja_jumat' not in dt_collector.columns:
            dt_collector['collector_kerja_jumat'] = dt_collector.apply(lambda row: self._check_collector_kerja_jumat(row), axis=1)
        if 'collector_kerja_sabtu' not in dt_collector.columns:
            dt_collector['collector_kerja_sabtu'] = dt_collector.apply(lambda row: self._check_collector_kerja_sabtu(row), axis=1)
        if 'collector_kerja_minggu' not in dt_collector.columns:
            dt_collector['collector_kerja_minggu'] = dt_collector.apply(lambda row: self._check_collector_kerja_minggu(row), axis=1)

        dt_collector.drop(['collector_workday_Sabtu-Minggu','collector_workday_Senin-Jumat', 'collector_workday_Senin-Minggu',
                       'collector_worktime_Pagi-Siang','collector_worktime_Pagi-Sore', 'collector_worktime_Siang-Malam',
                       'collector_worktime_Sore-Malam'], axis=1, inplace=True)
        return dt_collector

    def _adjust_education_level(self, dt_collector):
        # Add a new column with the value 0, so that the collector_as_dum has the same dimensions as the debtor_as_dum
        dt_collector['collector_education_level_SD'] = [0] * 1000
        dt_collector['collector_education_level_SMP'] = [0] * 1000

        # Determines the order of the new columns
        new_order = ['collector_age', 'collector_latitude', 'collector_longitude','collector_education_level_D3', 'collector_education_level_D4',
                    'collector_education_level_S1','collector_education_level_S2', 'collector_education_level_S3', 'collector_education_level_SD',
                    'collector_education_level_SMA', 'collector_education_level_SMP', 'collector_kerja_pagi', 'collector_kerja_siang',
                    'collector_kerja_sore', 'collector_kerja_malam', 'collector_kerja_senin', 'collector_kerja_selasa', 'collector_kerja_rabu',
                    'collector_kerja_kamis', 'collector_kerja_jumat', 'collector_kerja_sabtu', 'collector_kerja_minggu']

        # Use slicing to change the order of columns
        dt_collector = dt_collector[new_order]
        return dt_collector

    def _check_collector_kerja_pagi(self, row):
        return 1 if (row['collector_worktime_Pagi-Siang'] == 1) or (row['collector_worktime_Pagi-Sore'] == 1) else 0

    def _check_collector_kerja_siang(self, row):
        return 1 if (row['collector_worktime_Pagi-Siang'] == 1) or (row['collector_worktime_Pagi-Sore'] == 1) or (row['collector_worktime_Siang-Malam'] == 1) else 0

    def _check_collector_kerja_sore(self, row):
        return 1 if (row['collector_worktime_Pagi-Sore'] == 1) or (row['collector_worktime_Sore-Malam'] == 1) else 0

    def _check_collector_kerja_malam(self, row):
        return 1 if (row['collector_worktime_Siang-Malam'] == 1) or (row['collector_worktime_Sore-Malam'] == 1)  else 0

    def _check_collector_kerja_senin(self, row):
        return 1 if (row['collector_workday_Senin-Jumat'] == 1) or (row['collector_workday_Senin-Minggu'] == 1) else 0

    def _check_collector_kerja_selasa(self, row):
        return 1 if (row['collector_workday_Senin-Jumat'] == 1) or (row['collector_workday_Senin-Minggu'] == 1) else 0

    def _check_collector_kerja_rabu(self, row):
        return 1 if (row['collector_workday_Senin-Jumat'] == 1) or (row['collector_workday_Senin-Minggu'] == 1) else 0

    def _check_collector_kerja_kamis(self, row):
        return 1 if (row['collector_workday_Senin-Jumat'] == 1) or (row['collector_workday_Senin-Minggu'] == 1) else 0

    def _check_collector_kerja_jumat(self, row):
        return 1 if (row['collector_workday_Senin-Jumat'] == 1) or (row['collector_workday_Senin-Minggu'] == 1) else 0

    def _check_collector_kerja_sabtu(self, row):
        return 1 if (row['collector_workday_Sabtu-Minggu'] == 1) or (row['collector_workday_Senin-Minggu'] == 1) else 0

    def _check_collector_kerja_minggu(self, row):
        return 1 if (row['collector_workday_Sabtu-Minggu'] == 1) or (row['collector_workday_Senin-Minggu'] == 1) else 0

def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    radius = 6371  # Radius of the Earth in kilometers
    distance = radius * c
    # return distance between the two points in kilometers.
    return distance

def calculate_distance_per_collector(dt_debtor, dt_collector):
    dist = []
    for i in range(len(dt_collector)):
      dist.append(calculate_distance(dt_debtor["debtor_latitude"].values[0],
                                    dt_debtor["debtor_longitude"].values[0],
                                    dt_collector["collector_latitude"][i],
                                    dt_collector["collector_longitude"][i]))
    return dist

def scale_for_recsys(dt_debtor, dt_collector):
    dt_merge = pd.concat([dt_debtor, dt_collector], axis=1)
    features = ['debtor_age', 'debtor_latitude', 'debtor_longitude', 
                'collector_age', 'collector_latitude', 'collector_longitude']
    
    stdscaler = StandardScaler()

    for feature in features:
        dt_merge[feature] = stdscaler.fit_transform(dt_merge[[feature]])

    xdebt = dt_merge.iloc[:,:22]
    xcoll = dt_merge.iloc[:,22:]

    return xdebt, xcoll

class CalculateDistanceFromCoordinates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['distance'] = X.apply(lambda row: calculate_distance(row['collector_latitude'], row['collector_longitude'],
                                                                  row['debtor_latitude'], row['debtor_longitude']), axis=1)

        X.drop(['collector_latitude', 'collector_longitude', 'debtor_latitude', 'debtor_longitude' ], axis=1, inplace=True)
        return X


class TravelingDurationGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        interactions_df = X[['distance', 'departure_time',
                            'arrival_time', 'transportation_type', 'call_pickup_duration',
                            'door_opening_duration', 'connection_time', 'waiting_response_duration',
                            'idle_duration', 'nonproductive_duration']]
        fmt = '%H:%M'
        interactions_df['departure_time'] = interactions_df['departure_time'].apply(lambda x: datetime.strptime(x, fmt))
        interactions_df['arrival_time'] = interactions_df['arrival_time'].apply(lambda x: datetime.strptime(x, fmt))
        interactions_df['traveling_duration'] = (interactions_df['arrival_time'] - interactions_df['departure_time']).dt.total_seconds().astype(int)

        interactions_df.drop(['departure_time', 'arrival_time'], axis=1, inplace=True)
        return interactions_df

    
data_preprocessing_best_time = Pipeline([
    ('object_to_categories', ObjectToCategoriesDebtor()),
    ('debt_working_time_transformer', DebtorWorkingTimeTransformer()),
])

data_preprocessing_debtor = Pipeline([
    ('object_to_categories', ObjectToCategoriesDebtor()),
    ('preprocess_for_recsys', PreprocessDebtorForRecSys())
])

data_preprocessing_collector = Pipeline([
    ('object_to_categories', ObjectToCategoriesCollector()),
    ('preprocess_for_recsys', PreprocessCollectorForRecSys())
])

data_preprocessing_input_inter = Pipeline([
    ('calculate_distance_from_coordinates', CalculateDistanceFromCoordinates()),
    ('traveling_duration_generator', TravelingDurationGenerator()),
    ('categorical_encoder', CategoricalEncoder())
])

data_preprocessing_interactions = Pipeline([
    ('traveling_duration_generator', TravelingDurationGenerator()),
    ('categorical_encoder', CategoricalEncoder())
])

scale_and_pca = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('pca', PCA(n_components=3))
])