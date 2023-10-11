from ..apps import Fitur19Config
from ..libraries import utils

import pandas as pd

def predict_default_payment(data):
    model = Fitur19Config.default_payment_pred_model
    DATASET_FILE_NAME = 'debtor_dum.csv'

    input_df = transform_input(data)
    output = model.predict(input_df)
    result = transform_default_payment_pred_output(output)
    # result = output
    utils.append_dataset_with_new_data(DATASET_FILE_NAME, input_df, result)

    return result

def transform_input(data):
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)

    return df

def transform_default_payment_pred_output(pred):
    default_payment_next_month = pred[0]

    if default_payment_next_month == 0:
        default_payment_next_month = 'Not Default'
    elif default_payment_next_month == 1:
        default_payment_next_month = 'Default Payment'

    data = {
      'default_payment_next_month': default_payment_next_month,
  }

    return data