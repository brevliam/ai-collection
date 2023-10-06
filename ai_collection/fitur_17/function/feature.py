from ..apps import Fitur17Config
from ..libraries import utils

import pandas as pd

def predict_fraud_score(data):
  model = Fitur17Config.fraud_pred_model
  DATASET_FILE_NAME = '17_fraud_dummy_data_v01_041023.csv'

  input_df = transform_input(data)
  output = model.predict(input_df)
  result = transform_fraud_pred_output(output)

  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_remedial_score(data):
  model = Fitur17Config.remedial_pred_model
  DATASET_FILE_NAME = '17_remedial_dummy_data_v01_051023.csv'

  input_df = transform_input(data)
  output = model.predict(input_df)
  result = transform_remedial_pred_output(output)

  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def transform_input(data):
  data = {key: [value] for key, value in data.items()}
  df = pd.DataFrame(data)

  return df

def transform_fraud_pred_output(pred):
  fraud_score = pred[0]
  fraud_label = ''

  if fraud_score <= 250:
    fraud_label = 'No Fraud'
  elif fraud_score >= 500:
    fraud_label = 'Fraud'
  else:
    fraud_label = 'Suspect'

  data = {
    'fraud_score': fraud_score,
    'fraud_label': fraud_label
  }

  return data

def transform_remedial_pred_output(pred):
  remedial_score = pred[0]
  remedial_label = ''

  if remedial_score <= 700:
    remedial_label = 'No remedial'
  else:
    remedial_label = 'remedial'

  data = {
    'remedial_score': remedial_score,
    'remedial_label': remedial_label
  }

  return data
