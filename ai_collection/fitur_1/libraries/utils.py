"""
Module for data processing, prediction, and saving results to a CSV file.
"""

import os
import csv

def save_input_data_to_csv(input_data, collection_difficulty_score, collection_difficulty_category):
    """
    Save input data along with collection difficulty score and category to a CSV file.

    Parameters:
    - input_data (dict): Input data in the form of a dictionary.
    - collection_difficulty_score (float): Collection difficulty score.
    - collection_difficulty_category (str): Collection difficulty category.

    Returns:
    None
    """
    # Get the directory containing this Python script (libraries/utils.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the CSV file
    csv_file_path = os.path.join(script_dir, '..',
                                 'dataset', 'collection_difficulty_scoring_data.csv')

    column_names = [
        'debtor_nik', 'debtor_name', 'debtor_gender', 'debtor_birth_place',
        'debtor_age', 'debtor_zip', 'debtor_rt', 'debtor_rw', 'debtor_address',
        'debtor_number', 'bulan_1', 'bulan_2', 'bulan_3', 'bulan_4', 'bulan_5',
        'bulan_6', 'bulan_7', 'bulan_8', 'bulan_9', 'bulan_10', 'bulan_11',
        'bulan_12', 'loan_amount', 'tenor', 'billing_frequency',
        'transaction_frequency', 'interest_rate', 'other_credit', 'monthly_income',
        'home_ownership', 'employment_status', 'employment_status_duration',
        'public_records', 'contact_frequency', 'dti', 'revolving_credit_balance',
        'occupation', 'financial_situation', 'delay_history',
        'amount_of_late_days', 'number_of_dependents', 'number_of_vehicles',
        'number_of_complaints', 'external_factors',
        'communication_channel', 'working_time', 'payment_patterns', 'payment_method',
        'aging', 'collection_difficulty_score', 'collection_difficulty_category'
    ]

    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names)

        # Check if the file is empty and write the header if it is
        if csv_file.tell() == 0:
            writer.writeheader()

        # Fill in all columns with appropriate values from input_data
        input_data.update({
            'collection_difficulty_score': collection_difficulty_score,
            'collection_difficulty_category': collection_difficulty_category
        })
        writer.writerow(input_data)
