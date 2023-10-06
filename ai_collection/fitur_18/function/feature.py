# This module contains every method for inference, as well as data transformation and preprocessing.

from ..apps import Fitur18Config
from ..libraries import utils

import pandas as pd

def loss_reverse(data):
  model = Fitur18Config.loss_reverse
  DATASET_FILE_NAME = 'AI_Collection_and_Loss_Reverse_Forecast.csv'
  
  input_df = transform_input(data)
  output = model.predict(input_df)
  result = output
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
  
  return result

# def credit_risk(data):
#   model = Fitur18Config.credit_risk
#   DATASET_FILE_NAME = 'AI_Collection_and_Loss_Reverse_Forecast.csv'
  
#   input_df = transform_input(data)
#   output = model.predict(input_df)
#   result = output
#   utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
  
#   return result

# def time_to_collect(data):
#   model = Fitur18Config.time_to_collect
#   DATASET_FILE_NAME = 'AI_Collection_and_Loss_Reverse_Forecast.csv'
  
#   input_df = transform_input(data)
#   output = model.predict(input_df)
#   result = output
#   utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
  
#   return result

# def total_cost(data):
#   model = Fitur18Config.total_cost
#   DATASET_FILE_NAME = 'AI_Collection_and_Loss_Reverse_Forecast.csv'
  
#   input_df = transform_input(data)
#   output = model.predict(input_df)
#   result = output
#   utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
  
#   return result



def transform_input(data):
  data = {key: [value] for key, value in data.items()}
  df = pd.DataFrame(data)
  
  return df
