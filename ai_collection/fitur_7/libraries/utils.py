import pandas as pd
import json

def transform_workload_pred_input(data):
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
    return df

def transform_workload_pred_output(pred):
    workload_score = pred[0]
    workload_level = ''
    if workload_score < 500:
      workload_level = 'underload'
    elif workload_score > 700:
      workload_level = 'overload'
    else:
      workload_level = 'normal'

    data = {
        'workload_score': workload_score,
        'workload_level': workload_level
    }

    return data