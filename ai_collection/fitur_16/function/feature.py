# This module contains every method for inference, as well as data transformation and preprocessing.

from ..apps import Fitur16Config
from ..libraries import utils

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformers
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer



#kolom yang digunakan untuk prediksi
class Coll(BaseEstimator, TransformerMixin):

  def fit(self, df, y=None): 
    return self

  def transform(self, df):
    df = df.iloc[:, 4:]

    return df
  #df.drop(['collateral_auction_type', 'collateral_name', 'collateral_specification', "collateral_info"], axis = 1)
  
#Membuat outlier menjadi rata-rata
class OutlierMean(BaseEstimator, TransformerMixin):
  def fit(self, df, y=None): 
   return self

  def transform(self, df):

    #invoice  market price
    mean_valueMP = df['collateral_market_price'].mean()
    lower_limitMP = df['collateral_market_price'].quantile(0.05)
    upper_limitMP = df['collateral_market_price'].quantile(0.95)
    
    df['collateral_market_price'] =df['collateral_market_price'].apply(lambda x: mean_valueMP if x < lower_limitMP or x > upper_limitMP else x)
    # df = df[(df['invoice_auction_price'] > lower_limitMP) & (df['invoice_auction_price'] < upper_limitMP)]
    
    
    #invoice  auction price
    mean_valueAP = df['invoice_auction_price'].mean()
    lower_limitAP = df['invoice_auction_price'].quantile(0.05)
    upper_limitAP = df['invoice_auction_price'].quantile(0.95)

    df['invoice_auction_price'] = df['invoice_auction_price'].apply(lambda x: mean_valueAP if x < lower_limitAP or x > upper_limitAP else x)
    # df = df[(df['invoice_auction_price'] > lower_limitAP) & (df['invoice_auction_price'] < upper_limitAP)]

    return df
  
#Menginput dan menggunakan scaler untuk kolom numerik

# class FeatureScaling(BaseEstimator, TransformerMixin):
#   def fit(self, df, y=None): 
#    return self

#   def transform(self, df):
#     numeric_columns = ['add_income', 'collateral_market_price', 'invoice_auction_price', 'mrent_price', 'cc_bill', 'food_cost', 'trans_cost', 'employee_installment_loan', 'monthly_income', 'salary_reduction', 'Bonus_income', 'salary_reduction', 'dependents_1', 'dependents_2', 'dependents_3', 'dependents_4', 'dependents_5', 'coworker_report']
#     imputer = SimpleImputer(strategy="mean")
#     df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# #
#     scaler = StandardScaler()
#     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

#     return df 
  
# class FeatureScaling(BaseEstimator, TransformerMixin):
#   def fit(self, X, y=None): 
#     return self

#   def transform(self, X):
    
#     numeric_columns = ['add_income', 'collateral_market_price', 'invoice_auction_price', 'mrent_price', 'cc_bill', 'food_cost', 'trans_cost', 'employee_installment_loan', 'monthly_income', 'salary_reduction', 'Bonus_income', 'salary_reduction', 'dependents_1', 'dependents_2', 'dependents_3', 'dependents_4', 'dependents_5', 'coworker_report']
#     imputer = SimpleImputer(strategy="mean")
#     X[numeric_columns] = imputer.fit_transform(X[numeric_columns])
#     scaler = StandardScaler()
#     X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
#     return X
  


  
  # X['invoice_num'] = ~X['invoice_num'].isnull()
  # X['invoice_num']


class changetoCategorical(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
      #Mengubah invoice num menjadi boolean
       df['invoice_num'] = ~df['invoice_num'].isnull()
     #Mengubah invoice num dan invoice ttd menjadi category
      #  df['invoice_ttd'] = df['invoice_ttd'].astype('category')
      #  df['invoice_num'] = df['invoice_num'].astype('category')

       return df
    
class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
      

      if "True" in df['invoice_ttd']:
        df['invoice_ttd_0.0'] = 0
        df['invoice_ttd_1.0'] = 1
      else: 
        df['invoice_ttd_0.0'] = 1
        df['invoice_ttd_1.0'] = 0



      # Membuat kolom baru 'invoice_num_1.0' dengan nilai 1 jika 'invoice_num' tidak null, 0 jika null
      df['invoice_num_1.0'] = df['invoice_num'].notna().astype(int)
        # Membuat kolom baru 'invoice_num_0.0' dengan nilai 1 jika 'invoice_num' null, 0 jika tidak null
      df['invoice_num_0.0'] = df['invoice_num'].isna().astype(int)
      
        # Hapus kolom 'invoice_ttd' dan 'invoice_num' jika perlu
      df = df.drop(['invoice_ttd', 'invoice_num'], axis=1)
        

      return df
    
#class untuk Taksiran

categorical_columns = [
    "collateral_name", "Body", "paint", "glass", "tire", "mechine", "transmission", "suspension", "brake", "AC", "audio","power window",
    "power steering", "airbag", "ABS", "EBD", "ESP", "seat", "dashboard", "doortrim", "carpet", "interior lights","stang", "screen",
    "frame", "Remote control", "accessories", "contrast", "clarity", "refresh rate speed", "strength", "bass", "treble", "image color", "video color", "button", "port", "camera",
    "operating system", "processor", "RAM", "storage", "battery", "connectivity","lens", "viewfinder", "censor", "feature", "Detail",
    "resolution", "frame rate", "stability", "Keyboard", "mouse", "graphics card", "webcam", "mikrofon", "speaker", "strap", "Connector", "cable",
    ] 

class CollT(BaseEstimator, TransformerMixin):

  def fit(self, df, y=None): 
    return self

  def transform(self, df):
    # Menggunakan iloc untuk memilih kolom dengan indeks tertentu
    # df = df.iloc[:, list(range(0, 62)) + list(range(64, len(df.columns)))]
    df = df.drop(df.columns[63], axis=1)
    # df = df.iloc[:, 0:63]

    return df

class changetoCategoricalT(BaseEstimator, TransformerMixin):

  def fit(self, df, y=None): 
    return self

  def transform(self, df):

    for column in categorical_columns:
        df[column] = df[column].astype('category')

    return df


class changetoCategoricalT(BaseEstimator, TransformerMixin):
  def fit(self, df, y=None):
    return self

  def transform(self, df):
    
     #Mengubah invoice num dan invoice ttd menjadi category

    df[categorical_columns] = df[categorical_columns].astype('category')

    return df

    


class FeatureEncoderT(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
       return self

    def transform(self, df):
      
        imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

        # Melakukan label t encoding pada kolom kategori

        # label_encoders = {}
        for col in categorical_columns:
          le = LabelEncoder()
          df[col] = le.fit_transform(df[col])
          
        return df

def pipeT(df):
   pipeline = Pipeline([
      ("droper", CollT()),
      ("changetoCategorical", changetoCategoricalT()),
      # ("handlingoutlier", OutlierMeanT()),
      # ("Scaling", FeatureScalingT()),
      ("encoder", FeatureEncoderT()) 
])
   transform_data = pipeline.fit_transform(df)
   return transform_data


def create_dataframe(data):
    df = {key: [value] for key, value in data.items()}
    return pd.DataFrame(df)

def pipe(df):
   
   pipeline = Pipeline([
    ("droper", Coll()), 
    ("handlingoutlier", OutlierMean()),
    # ("Scaling", FeatureScaling()),
    ("changetoCategorical", changetoCategorical()),
    ("encoder", FeatureEncoder()) 
])
   
   transform_data = pipeline.fit_transform(df)  
   return transform_data

def input_df(data):
    df = create_dataframe(data)
    
    # transformed_df = pipe(df)
    return df

def predict_fraud_category(data):
    model = Fitur16Config.fraud_prediction_model
    DATASET_FILE_NAME = 'data_Fraud.csv'
    
    input_d =  input_df(data)
    trans_in = transform_input(data)
    output = model.predict(trans_in)
    result = transform_fraud_pred_output(output)
    utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_d, result)
  
    return result

def predict_appraisal_score(data):
    model = Fitur16Config.appraisal_prediction_model 
    
    DATASET_FILE_NAME = 'collateral_appraisal.csv'
    
    input_d =  input_df(data)
    trans_in = transform_inputT(data)
    output = model.predict(trans_in)
    result = transform_collateral_pred_output(output)
    utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_d, result)
  
    return result

def transform_input(data):
    df = create_dataframe(data)
    
    transformed_df = pipe(df)
    return transformed_df

def transform_inputT(data):
    df = create_dataframe(data)
    
    transformed_df = pipeT(df)
    return transformed_df

def transform_fraud_pred_output(pred):
    data = {
        "fraud": pred,
    }
    return data


def transform_collateral_pred_output(pred):
    data = {
        "collateral_appraisal":round( pred[0]),
    }
    return data




