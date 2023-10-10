# This module contains every method for inference, as well as data transformation and preprocessing.

from ..apps import Fitur18Config
from ..libraries import utils
from rest_framework import status 
import json
import pandas as pd

      
class Prediction:
    def loss_reverse(self,request):
        return_dict=dict()
        try:
            input_request=request.body 
            decode_input_request=input_request.decode('utf8').replace("'",'"')
            request_dict=json.loads(decode_input_request)
            df_pred=pd.json_normalize(request_dict)

            model = Fitur18Config.loss_reverse
            model2 = Fitur18Config.credit_risk
            prediction = model.predict(df_pred)
            prediction = round(prediction[0])
            df_pred["loss_reverse"] = prediction

            prediction2 = model2.predict(df_pred)
            DATASET_FILE_NAME = 'AI_Collection_and_Loss_Reverse_Forecast.csv'
            input_df = pd.DataFrame(df_pred)
            input_df.drop("loss_reverse", axis=1, inplace=True)
            input_df["credit_risk"] = prediction2[0]
            X = {"loss_reverse": prediction}
            utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, X)
            predictions = {
              "credit_risk": prediction2[0],
              "loss_reverse": prediction
            }
            return predictions

        except Exception as e: 
            return_dict['response']="Exception when prediction: "+str(e)
            return_dict['status']=status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict
          
          
    def time_to_collect(self,request):
        return_dict=dict()
        try:
            input_request=request.body 
            decode_input_request=input_request.decode('utf8').replace("'",'"')
            request_dict=json.loads(decode_input_request)
            df_pred=pd.json_normalize(request_dict)

            model = Fitur18Config.time_to_collect
            prediction = model.predict(df_pred)
            prediction = round(prediction[0])

            DATASET_FILE_NAME = 'kolektor.csv'
            input_df = pd.DataFrame(df_pred)
            X = {"loss_reverse": prediction}
            utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, X)
            return prediction

        except Exception as e: 
            return_dict['response']="Exception when prediction: "+str(e)
            return_dict['status']=status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict
          
    def total_cost(self,request):
        return_dict=dict()
        try:
            input_request=request.body 
            decode_input_request=input_request.decode('utf8').replace("'",'"')
            request_dict=json.loads(decode_input_request)
            df_pred=pd.json_normalize(request_dict)

            model = Fitur18Config.total_cost
            prediction = model.predict(df_pred)
            prediction = round(prediction[0])

            DATASET_FILE_NAME = 'kolektor.csv'
            input_df = pd.DataFrame(df_pred)
            X = {"loss_reverse": prediction}
            utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, X)
            return prediction
          
        except Exception as e: 
            return_dict['response']="Exception when prediction: "+str(e)
            return_dict['status']=status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict
          

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




