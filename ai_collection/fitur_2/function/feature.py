# This module contains every method for inference, as well as data transformation and preprocessing.
from ..apps import Fitur2Config
from ..libraries import utils
import pandas as pd

def predict_debtor_label_by_age(data):
  model = Fitur2Config.debtor_class_by_age
  DATASET_FILE_NAME = 'Dummy Data Debitur_v06_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_debtor_label_by_location(data):
  model = Fitur2Config.debtor_class_by_location
  DATASET_FILE_NAME = 'Dummy Data Debitur_v06_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_debtor_label_by_behavior(data):
  model = Fitur2Config.debtor_class_by_behavior
  DATASET_FILE_NAME = 'Dummy Data Debitur_v06_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_debtor_label_by_character(data):
  model = Fitur2Config.debtor_class_by_character
  DATASET_FILE_NAME = 'Dummy Data Debitur_v06_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_debtor_label_by_collector_field(data):
  model = Fitur2Config.debtor_class_by_collector_field
  DATASET_FILE_NAME = 'Dummy Data Debitur_v06_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_debtor_label_by_SES(data):
  model = Fitur2Config.debtor_class_by_ses
  DATASET_FILE_NAME = 'Dummy Data Debitur_v06_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_debtor_label_by_demography(data):
  model = Fitur2Config.debtor_class_by_demography
  DATASET_FILE_NAME = 'Dummy Data Debitur_v06_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_collector_label_by_age(data):
  model = Fitur2Config.collector_class_by_age
  DATASET_FILE_NAME = 'Dummy Data Kolektor_v04_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_collector_label_by_location(data):
  model = Fitur2Config.collector_class_by_location
  DATASET_FILE_NAME = 'Dummy Data Kolektor_v04_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_collector_label_by_behavior(data):
  model = Fitur2Config.collector_class_by_behavior
  DATASET_FILE_NAME = 'Dummy Data Kolektor_v04_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_collector_label_by_character(data):
  model = Fitur2Config.collector_class_by_character
  DATASET_FILE_NAME = 'Dummy Data Kolektor_v04_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_collector_label_by_collector_field(data):
  model = Fitur2Config.collector_class_by_collector_field
  DATASET_FILE_NAME = 'Dummy Data Kolektor_v04_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_collector_label_by_SES(data):
  model = Fitur2Config.collector_class_by_ses
  DATASET_FILE_NAME = 'Dummy Data Kolektor_v04_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def predict_collector_label_by_demography(data):
  model = Fitur2Config.collector_class_by_demography
  DATASET_FILE_NAME = 'Dummy Data Kolektor_v04_20231008.csv'
  
  input_df = transform_input(data)
  result = model.predict(input_df)
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

  return result

def transform_input(data):
  data = {key: [value] for key, value in data.items()}
  df = pd.DataFrame(data)

  return df


