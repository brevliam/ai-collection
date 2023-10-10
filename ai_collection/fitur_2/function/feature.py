# This module contains every method for inference, as well as data transformation and preprocessing.
from ..apps import Fitur2Config
from ..libraries import utils
import pandas as pd

def predict_debtor_label(data):
  model1 = Fitur2Config.debtor_class_by_age
  model2 = Fitur2Config.debtor_class_by_location
  model3 = Fitur2Config.debtor_class_by_behavior
  model4 = Fitur2Config.debtor_class_by_character
  model5 = Fitur2Config.debtor_class_by_collector_field
  model6 = Fitur2Config.debtor_class_by_ses
  model7 = Fitur2Config.debtor_class_by_demography
  
  DATASET_FILE_NAME = 'Dummy Data Debitur_v06_20231008.csv'
  
  input_df = transform_input(data)
  result1 = model1.predict(input_df)
  result2 = model2.predict(input_df)
  result3 = model3.predict(input_df)
  result4 = model4.predict(input_df)
  result5 = model5.predict(input_df)
  result6 = model6.predict(input_df)
  result7 = model7.predict(input_df)

  combined_results = pd.DataFrame({
        'age_label': result1,
        'location_label': result2,
        'behavior_label': result3,
        'character_label': result4,
        'collector_field_label': result5,
        'ses_label': result6,
        'demography_label': result7
  })

  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, combined_results)

  return combined_results

def predict_collector_label(data):
  model1 = Fitur2Config.collector_class_by_age
  model2 = Fitur2Config.collector_class_by_location
  model3 = Fitur2Config.collector_class_by_behavior
  model4 = Fitur2Config.collector_class_by_character
  model5 = Fitur2Config.collector_class_by_collector_field
  model6 = Fitur2Config.collector_class_by_ses
  model7 = Fitur2Config.collector_class_by_demography
  
  DATASET_FILE_NAME = 'Dummy Data Kolektor_v04_20231008.csv'
  
  input_df = transform_input(data)
  result1 = model1.predict(input_df)
  result2 = model2.predict(input_df)
  result3 = model3.predict(input_df)
  result4 = model4.predict(input_df)
  result5 = model5.predict(input_df)
  result6 = model6.predict(input_df)
  result7 = model7.predict(input_df)

  combined_results = pd.DataFrame({
        'age_label': result1,
        'location_label': result2,
        'behavior_label': result3,
        'character_label': result4,
        'collector_field_label': result5,
        'ses_label': result6,
        'demography_label': result7
  })

  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, combined_results)

  return combined_results

def transform_input(data):
  data = {key: [value] for key, value in data.items()}
  df = pd.DataFrame(data)

  return df


