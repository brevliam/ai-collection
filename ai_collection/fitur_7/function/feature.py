# This module contains every method for inference, as well as data transformation and preprocessing.

from ..apps import Fitur7Config
from ..libraries import utils

import pandas as pd

def predict_workload_score(data):
  model = Fitur7Config.workload_pred_model
  DATASET_FILE_NAME = 'workload_prediction_v3_230927.csv'
  
  input_df = transform_input(data)
  output = model.predict(input_df)
  result = transform_workload_pred_output(output)
  
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
  
  return result

def transform_input(data):
  data = {key: [value] for key, value in data.items()}
  df = pd.DataFrame(data)
  
  return df

def transform_workload_pred_output(pred):
  workload_score = pred[0]
  if workload_score > 1000:
      workload_score = 1000.0
  elif workload_score < 0:
      workload_score == 0.0

  workload_level = ''
  if workload_score < 500:
    workload_level = 'underload'
  elif workload_score > 700:
    workload_level = 'overload'
  else:
    workload_level = 'normal'

  data = {
      'workload_score': workload_score,
      'workload_level': workload_level
  }

  return data