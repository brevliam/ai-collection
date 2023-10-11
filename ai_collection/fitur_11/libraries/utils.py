# This module contains every method for appending new data to the dataset.

import pandas as pd
import os

def load_dataset_path(filename):
   feature_dir = os.path.dirname(os.path.abspath(__file__))
   dataset_dir = os.path.join(feature_dir, "..", "dataset")
   file_path = os.path.join(dataset_dir, filename)
   
   return file_path

def append_new_row(dataset_file_name,df):
    dataset_path = load_dataset_path(dataset_file_name)
    with open(dataset_path, 'a', newline='') as file:
        df.to_csv(file, header=False, index=False)