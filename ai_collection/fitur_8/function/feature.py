# This module contains every method for inference, as well as data transformation and preprocessing.

from ..apps import Fitur8Config
from ..libraries import utils

import pandas as pd

def predict_Efficient_HumanResources_score(data):
  model = Fitur8Config.Effecientcosteffectiveness_model
  DATASET_FILE_NAME = 'AIcollection_costeffectiveness_terbaruu.csv'
  
  input_df = transform_input(data)
  output = model.predict(input_df)
  result = transform_Effecientcosteffectiveness_pred_output(output)
  
  utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)
  
  return result



def transform_input(data):
  data = {key: [value] for key, value in data.items()}
  df = pd.DataFrame(data)
  
  return df

def transform_Effecientcosteffectiveness_pred_output(pred):
  Effecient_HumanResources_score = pred[0]
  Effecient_HumanResources_label = ''
  
  if Effecient_HumanResources_score == 0:
    Effecient_HumanResources_label = 'Efficiency & Effectiveness'
  elif Effecient_HumanResources_score >=3:
    Effecient_HumanResources_label = 'Efficiency & Effectiveness'  
  else:
   Effecient_HumanResources_label = 'NO Efficiency & Effectiveness'

  data = {
      'Effecientcosteffectiveness_score': Effecient_HumanResources_score,
      'Effecientcosteffectiveness_label': Effecient_HumanResources_label
  }

  return data

