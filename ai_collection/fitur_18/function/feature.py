# This module contains every method for inference, as well as data transformation and preprocessing.

from ..apps import Fitur18Config
from ..libraries import utils
from rest_framework import status 
import json
import pandas as pd

class Prediction:
    def predict(self,request):
        return_dict=dict()
        try:
            input_request=request.body 
            decode_input_request=input_request.decode('utf8').replace("'",'"')
            request_dict=json.loads(decode_input_request)
            df_pred=pd.json_normalize(request_dict)

            model = Fitur18Config.loss_reverse
            prediction=model.predict(df_pred)
            print(prediction)

            request_dict['prediction']=prediction 
            return_dict['response']=request_dict
            return_dict['status']=status.HTTP_200_OK
            return prediction

        except Exception as e: 
            return_dict['response']="Exception when prediction: "+str(e)
            return_dict['status']=status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict  
          
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
