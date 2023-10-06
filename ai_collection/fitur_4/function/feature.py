from ..apps import Fitur4Config
from ..libraries import utils
from ..apps import Fitur4Config2
import pandas as pd
import numpy as np

def predict_assignment(data):
    model = Fitur4Config.assignment_pred_model
    DATASET_FILE_NAME = 'df_assignment.csv'
    
    input_df = transform_input(data)

    output = model.predict(input_df)
    # result = transform_workload_pred_output(output)
    output_model = output[0]
    output_df = output
    # utils.append_dataset_with_new_data_assignment(DATASET_FILE_NAME, input_df, output_df)
    
    return output_df

def predict_campaign(data):
    model = Fitur4Config2.campaign_pred_model
    DATASET_FILE_NAME = 'df_campaign.csv'
    
    input_df = transform_input(data)

    output = model.predict(input_df)
    output_model = Pred_cluster_campaign(output,input_df)
    output_df = np.array([output_model])
    utils.append_dataset_with_new_data_camapign(DATASET_FILE_NAME, input_df, output_df)
    
    return output_model

def transform_input(data):
    # data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame([data])
    return df


def Pred_cluster_campaign(predictions,data):
    var_vis = 'total_visit'
    foreclosure = 'foreclosure'


    visit_campaign_cont = 2
    visit_campaign_beg = 0
    call_campaign = 1

    predictions = predictions[0]


    


    # Predict New data


    campaign_selected = None  # Initialize campaign_selected

    if predictions == visit_campaign_cont:
        if data[var_vis].values[0] > 4:
            campaign_selected = 'Lakukan kunjungan Ulang'
        else:
            if data[foreclosure].values[0] == False:
                campaign_selected = 'Lakukan Penyitaan Barang'
            elif data[foreclosure].values[0] == True:
                campaign_selected = 'Lakukan Penutupan buku Kreditur'

    elif predictions == visit_campaign_beg:
        campaign_selected = 'Lakukan kunjungan Pertama'

    elif predictions == call_campaign:
        campaign_selected = 'Lakukan Telfon'

    if campaign_selected is None:
        campaign_selected = 'No campaign selected'

    return campaign_selected

