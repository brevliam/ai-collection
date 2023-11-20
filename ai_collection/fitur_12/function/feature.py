"""
Module: feature
Description: Contains API feature calculate predictions.
"""
import pandas as pd
from ..apps import Fitur12Config
from ..libraries import utils

def calculate_monthly_installments(loan_amount, interest_rate, tenor):
    """
    Calculate mothly installments.

    Parameters:
        data : loan_amount, interest_rate, tenor.

    Returns:
        float: monthly installments.
    """
    return ((loan_amount+(loan_amount*interest_rate/100))/tenor)

def predict_tenor(data):
    """
    Predicts the tenor based on input data.

    Parameters:
        data (dict): Input data for prediction.

    Returns:
        dict: Response containing debtor information, recommended tenor, and monthly payments.
    """
    model = Fitur12Config.tenor_pred_model
    dataset_file_name = 'Dataset Recommendation Tenor and Monthly Payments 05102023.csv'

    mydata=[data]
    input_df = pd.DataFrame(mydata)
    #input_df = transform_input(data)
    y_pred = round(model.predict(input_df)[0])
    select_response = ['debtor_name', 'debtor_nik', 'debtor_id']
    df_response = input_df[select_response]
    monthly_installment = round(calculate_monthly_installments(input_df['loan_amount'], input_df['interest_rate'], y_pred), 2)
    # Menambahkan kolom prediksi tenor dan perhitungan pembayaran bulanan untuk output json
    df_response = df_response.assign(recomendation_tenor=y_pred, recomendation_monthly_payments=monthly_installment)
    dict_response = df_response.to_dict(orient='records')
    # Membuat data baru dan mengganti nilai Pembayaran bulanan
    #input_df = input_df.drop('monthly_payments', axis=1)
    data_append = {
      "tenor" : y_pred,
      "monthly_payments" : monthly_installment[0]
    }
    utils.append_dataset_next_row(dataset_file_name, input_df, data_append)
    return dict_response

def predict_loan(data):
    """
    Predicts the loan amount based on input data.

    Parameters:
        data (dict): Input data for prediction.

    Returns:
        dict: Response containing debtor information and the predicted loan amount.
    """
    model = Fitur12Config.loan_pred_model
    dataset_file_name = 'Dataset Request Loan 05102023.csv'

    mydata=[data]
    input_df = pd.DataFrame(mydata)
    #input_df = transform_input(data)
    y_pred = round(model.predict(input_df)[0])
    select_response = ['debtor_name', 'debtor_nik', 'debtor_id']
    df_response = input_df[select_response]
    # Menambahkan kolom prediksi tenor dan perhitungan pembayaran bulanan
    df_response = df_response.assign(request_loan=y_pred)
    dict_response = df_response.to_dict(orient='records')
    data_append = {
      "loan_amount" : y_pred,
    }
    utils.append_dataset_next_row(dataset_file_name, input_df, data_append)
    return dict_response

def transform_input(data):
    """
    Transforms input data into a DataFrame.

    Parameters:
        data (dict): Input data.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
    return df
  