# This module contains every method for inference, as well as data transformation and preprocessing.
from ..apps import Fitur12Config
from ..libraries import utils
import pandas as pd

def calculate_monthly_installments(loan_amount, interest_rate, tenor):
  return ((loan_amount+(loan_amount*interest_rate/100))/tenor)

def predict_tenor(data):
  model = Fitur12Config.tenor_pred_model
  DATASET_FILE_NAME = 'Dataset AI Restructure Approval rev7 05102023.csv'

  mydata=[data]
  input_df = pd.DataFrame(mydata)
  #input_df = transform_input(data)
  y_pred=model.predict(input_df)[0]
  select_response = ['debtor_name', 'debtor_nik', 'debtor_id',
					 		'debtor_occupation']
  df_response = input_df[select_response]
  monthly_installment = calculate_monthly_installments(input_df['loan_amount'], input_df['interest_rate'], y_pred)
	# Menambahkan kolom prediksi tenor dan perhitungan pembayaran bulanan untuk output json
  df_response = df_response.assign(recomendation_tenor=y_pred, recomendation_monthly_payments=monthly_installment)
  dict_response = df_response.to_dict(orient='records')
  # Membuat data baru dan mengganti nilai Pembayaran bulanan
  #input_df = input_df.drop('monthly_payments', axis=1)
  data_append = {
    "tenor" : y_pred,
    "monthly_payments" : monthly_installment[0]
  }
  utils.append_dataset_recommendation_tenor(DATASET_FILE_NAME, input_df, data_append)
  return dict_response

def predict_loan(data):
  model = Fitur12Config.loan_pred_model
  DATASET_FILE_NAME = 'Dataset AI Restructure Approval rev7 05102023.csv'

  mydata=[data]
  input_df = pd.DataFrame(mydata)
  #input_df = transform_input(data)
  y_pred=model.predict(input_df)[0]
  select_response = ['debtor_name', 'debtor_nik', 'debtor_id',
					 		'debtor_occupation']
  df_response = input_df[select_response]
	# Menambahkan kolom prediksi tenor dan perhitungan pembayaran bulanan
  df_response = df_response.assign(request_loan=y_pred)
  dict_response = df_response.to_dict(orient='records')
  data_append = {
    "loan_amount" : y_pred,
  }
  utils.append_dataset_request_loan(DATASET_FILE_NAME, input_df, data_append)
  return dict_response

def transform_input(data):
  data = {key: [value] for key, value in data.items()}
  df = pd.DataFrame(data)
  return df