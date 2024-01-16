# This module contains every method for inference, as well as data transformation and preprocessing.

from ..apps import Fitur20Config
from ..libraries import utils

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformers
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder



class Coll(BaseEstimator, TransformerMixin):

  def fit(self, df, y=None): 
    return self

  def transform(self, df):
    df = df.iloc[:, :]

    return df
  
class FeatureScaling(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None): 
    return self

  def transform(self, X):
    
    numeric_columns = ["debtor_age", "debtor_number_of_dependents", "debtor_asset_value", "debtor_yearly_income", "debtor_yearly_expense", "debtor_net_income", "debtor_income_frequency", "debtor_loan_amount", "debtor_reschedule_history", "debtor_delay_history", "debtor_tenor", "debtor_num_of_paid_months", "debtor_num_of_unpaid_months", "debtor_days_before_due_date", "debtor_business_prospect_market_conditions", "debtor_performance_profitability", "jan","feb","mar","apr", "may", "jun", "jul", "august", "sept", "oct", "nov", "des"]
    imputer = SimpleImputer(strategy="mean")
    X[numeric_columns] = imputer.fit_transform(X[numeric_columns])
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
    return X
class FeatureEncoderT(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        # One-hot encoding and mapping for specified columns
        df["debtor_gender"] = df["debtor_gender"].map({"Laki-laki": 0, "Perempuan": 1})
        df["debtor_education_level"] = df["debtor_education_level"].map({"SD": 0, "SMP": 1, "SMA": 2, "D3": 3, "S1": 4, "S2": 5, "S3": 6})
        df["debtor_marital_status"] = df["debtor_marital_status"].map({"Belum Kawin": 0, "Kawin": 1, "Cerai Mati": 2, "Cerai Hidup": 3})
        # df["debtor_asset_ownership"] = df["debtor_asset_ownership"].astype(int)
        if "TRUE" in df['debtor_asset_ownership']:
           df['debtor_asset_ownership'] = 1
        else:
           df['debtor_asset_ownership'] = 0
        df["debtor_occupation"] = df["debtor_occupation"].map({"Buruh": 0, "Pegawai Negeri": 1, "Pegawai Swasta": 2, "Profesional": 3, "Pengusaha": 4})
        df["debtor_communication_channel"] = df["debtor_communication_channel"].map({"Telpon": 1, "SMS": 2, "E-mail": 3, "WhatsApp": 4})
        
        # df["debtor_business_prospect_growth_potential"] = df["debtor_business_prospect_growth_potential"].astype(int)
        if "TRUE" in df['debtor_business_prospect_growth_potential']:
           df['debtor_business_prospect_growth_potential'] = 1
        else:
           df['debtor_business_prospect_growth_potential'] = 0
        df["debtor_business_prospect_market_conditions"] = df["debtor_business_prospect_market_conditions"].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4})
        df["debtor_performance_profitability"] = df["debtor_performance_profitability"].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4})
        # df["debtor_performance_cash_flow"] = df["debtor_performance_cash_flow"].astype(int)
        if "TRUE" in df['debtor_performance_cash_flow']:
           df['debtor_performance_cash_flow'] = 1
        else:
           df['debtor_performance_cash_flow'] = 0

        # df["debtor_repayment_ability_accuracy"] = df["debtor_repayment_ability_accuracy"].astype(int)
        if "TRUE" in df['debtor_repayment_ability_accuracy']:
           df['debtor_repayment_ability_accuracy'] = 1
        else:
           df['debtor_repayment_ability_accuracy'] = 0

        df["debtor_aging"] = df["debtor_aging"].map({"Lancar": 0, "DPK": 1, "Kurang lancar": 2, "Diragukan": 3, "Macet": 4})

        categorical_columns = ["debtor_gender","debtor_education_level",'debtor_marital_status','debtor_asset_ownership','debtor_occupation','debtor_communication_channel','debtor_business_prospect_growth_potential','debtor_business_prospect_market_conditions', 'debtor_performance_profitability', 'debtor_performance_cash_flow', 'debtor_repayment_ability_accuracy', 'debtor_aging']

        imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

        # df["debtor_risk_status"] =  df["debtor_risk_status"].map({"Tidak Beresiko": 0, "Cukup Beresiko": 1, "Beresiko": 2, "Sangat Beresiko": 3})

        return df

def create_dataframe(data):
    df = {key: [value] for key, value in data.items()}
    return pd.DataFrame(df)

def pipe(df):  
   pipeline = Pipeline([   
    ("coll", Coll()),
    ("Scaling", FeatureScaling()),
    ("encoder", FeatureEncoderT())
])
   
   transform_data = pipeline.fit_transform(df)  
   return transform_data

def input_df(data):
    df = create_dataframe(data)
    return df


def risk_status(data):
   model = Fitur20Config.risk_status_prediction_model 
   DATASET_FILE_NAME = 'Remedial_analytics_V03_20231011.csv'
   input_d = input_df(data)
   
   trans_in = transform_input(data)
   output = model.predict(trans_in)
   result = transform_risk_status_output(output)
   utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_d, result)
  
   return result

def transform_input(data):
    df = create_dataframe(data)
    transformed_df = pipe(df)
    return transformed_df

def transform_risk_status_output(pred):
    data = {
        "debtor_risk_status":round( pred[0]),
    }
    return data




