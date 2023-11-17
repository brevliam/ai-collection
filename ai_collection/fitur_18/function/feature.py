"""
This module contains every method for inference, as well as data transformation and preprocessing.
"""

import json
from rest_framework import status
import pandas as pd
from ..apps import Fitur18Config
from ..libraries import utils


class Prediction:
    """
    This class contains methods for making predictions related to feature 18.
    """

    def loss_reverse(self, request):
        """
        Predicts loss reverse and credit risk based on the provided request.

        Parameters:
            - request (HttpRequest): The HTTP request containing input data.

        Returns:
            dict: A dictionary containing the predicted credit risk and loss reverse.
        """
        return_dict = {}
        try:
            df_pred = self._preprocess_data(request)

            model = Fitur18Config.loss_reverse
            model2 = Fitur18Config.credit_risk
            prediction = model.predict(df_pred)
            prediction = round(prediction[0])
            df_pred["loss_reverse"] = prediction

            prediction2 = model2.predict(df_pred)
            file_name = 'AI_Collection_and_Loss_Reverse_Forecast.csv'
            self._update_dataset(file_name, df_pred, prediction2[0])

            predictions = {"credit_risk": prediction2[0], "loss_reverse": prediction}
            return predictions

        except Exception as e:
            return_dict['response'] = f"Exception when prediction: {str(e)}"
            return_dict['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict

    def time_to_collect(self, request):
        """
        Predicts time to collect based on the provided request.

        Parameters:
            - request (HttpRequest): The HTTP request containing input data.

        Returns:
            int: The predicted time to collect.
        """
        return_dict = {}
        try:
            df_pred = self._preprocess_data(request)

            model = Fitur18Config.time_to_collect
            prediction = model.predict(df_pred)
            prediction = round(prediction[0])

            file_name = 'kolektor.csv'
            self._update_dataset(file_name, df_pred, prediction)

            return prediction

        except Exception as e:
            return_dict['response'] = f"Exception when prediction: {str(e)}"
            return_dict['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict

    def total_cost(self, request):
        """
        Predicts total cost based on the provided request.

        Parameters:
            - request (HttpRequest): The HTTP request containing input data.

        Returns:
            int: The predicted total cost.
        """
        return_dict = {}
        try:
            df_pred = self._preprocess_data(request)

            model = Fitur18Config.total_cost
            prediction = model.predict(df_pred)
            prediction = round(prediction[0])

            file_name = 'kolektor.csv'
            self._update_dataset(file_name, df_pred, prediction)

            return prediction

        except Exception as e:
            return_dict['response'] = f"Exception when prediction: {str(e)}"
            return_dict['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict

    def _preprocess_data(self, request):
        """
        Preprocesses the input data from the request.

        Parameters:
            - request (HttpRequest): The HTTP request containing input data.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        input_request = request.body
        decode_input_request = input_request.decode('utf8').replace("'", '"')
        request_dict = json.loads(decode_input_request)
        return pd.json_normalize(request_dict)

    def _update_dataset(self, file_name, input_df, prediction):
        """
        Updates the dataset with new data.

        Parameters:
            - file_name (str): The name of the dataset file.
            - input_df (pd.DataFrame): The input DataFrame.
            - prediction (int): The prediction to be added to the DataFrame.
        """
        input_df.drop("loss_reverse", axis=1, inplace=True)
        input_df["credit_risk"] = prediction
        loss_reverse = {"loss_reverse": prediction}
        utils.append_dataset_with_new_data(file_name, input_df, loss_reverse)
        