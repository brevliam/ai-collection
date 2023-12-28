"""
This module contains every method for inference, as well as data transformation and preprocessing.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from ..libraries import utils
from ..apps import Fitur13Config

def predict_best_time_to_remind(data):
    """
    Predict the best time to remind based on the input data.

    Parameters:
    - data (dict): Input data dictionary.

    Returns:
    dict: Prediction result.
    """
    model = Fitur13Config.best_time_to_remind_model
    dataset_file_name = 'Update_Data_Reminder.csv'
    input_df = transform_input_debtor(data)
    preprocessed_df = data_preprocessing_best_time_reminder.fit_transform(input_df)
    output = model.predict(preprocessed_df)
    output_proba = model.predict_proba(preprocessed_df).flatten()
    result = transform_best_time_to_remind_pred_output(output, output_proba)
    utils.append_dataset_with_new_data(dataset_file_name, input_df, result)
    return result

def predict_best_time_to_follow_up(data):
    """
    Predict the best time to follow up based on the input data.

    Parameters:
    - data (dict): Input data dictionary.

    Returns:
    dict: Prediction result.
    """
    model = Fitur13Config.best_time_to_follow_up_model
    dataset_file_name = 'Follow_Up_AI_Reschedule_Automation_V07_20231004.xlsx'
    input_df = transform_input_debtor(data)
    preprocessed_df = data_preprocessing_best_time_follow_up.fit_transform(input_df)
    # Drop instances where debtor_aging is 1
    preprocessed_df = preprocessed_df[preprocessed_df['debtor_aging'] != 1]
    output = model.predict(preprocessed_df)
    output_proba = model.predict_proba(preprocessed_df).flatten()
    result = transform_best_time_to_follow_up_pred_output(output, output_proba)
    utils.append_dataset_with_new_data(dataset_file_name, input_df, result)
    return result

def predict_reschedule(data):
    """
    Predict rescheduling eligibility based on the input data.

    Parameters:
    - data (dict): Input data dictionary.

    Returns:
    dict: Prediction result.
    """
    model = Fitur13Config.reschedule_model
    dataset_file_name = 'AI_Reschedule_data_terbaru.xlsx'
    input_df = transform_input_debtor(data)
    output = model.predict(input_df)
    result = transform_reschedule(output)
    utils.append_dataset_with_new_data(dataset_file_name, input_df, result)
    return result

def transform_input_debtor(data):
    """
    Transform input debtor data into a DataFrame.

    Parameters:
    - data (dict): Input data dictionary.

    Returns:
    pd.DataFrame: Transformed DataFrame.
    """
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
    return df

def transform_reschedule(output):
    """
    Transform rescheduling output into a result dictionary.

    Parameters:
    - output (array): Model output array.

    Returns:
    dict: Transformed result.
    """
    label_mapping = {
        0: 'Layak Reschedule Dengan Resiko Rendah',
        1: 'Layak Reschedule Dengan Resiko Sedang',
        2: 'Tidak Layak Reschedule Dengan Resiko Sedang',
        3: 'Tidak Layak Reschedule Dengan Resiko Tinggi'
    }
    reschedule_eligibility = label_mapping[output[0]]

    if reschedule_eligibility == label_mapping[0]:
        reshedule_context = (
            "Nasabah memiliki usia yang relatif muda, jumlah tanggungan yang rendah, aset yang mencukupi, pendapatan yang stabil dan tinggi, serta riwayat reschedule yang rendah."
        )
    elif reschedule_eligibility == label_mapping[1]:
        reshedule_context = (
            "Nasabah memiliki usia yang relatif muda, pendapatan yang stabil dan tinggi, serta riwayat reschedule yang rendah. Namun, nasabah memiliki jumlah tanggungan yang tinggi dan aset yang kurang mencukupi."
        )
    elif reschedule_eligibility == label_mapping[2]:
        reshedule_context = (
            "Nasabah memiliki usia yang relatif tua, jumlah tanggungan yang tinggi, aset yang kurang mencukupi, pendapatan yang tidak stabil, serta riwayat reschedule yang tinggi."
        )
    elif reschedule_eligibility == label_mapping[3]:
        reshedule_context = (
            "Nasabah memiliki usia yang relatif tua, jumlah tanggungan yang tinggi, aset yang kurang mencukupi, pendapatan yang tidak stabil, serta riwayat reschedule yang sangat tinggi."
        )
    else:
        reshedule_context = "Tidak termasuk dalam pertimbangan kategori reschedule."
    result = {
        'reschedule_eligibility': reschedule_eligibility,
        'reshedule_context': reshedule_context,
    }
    return result

def transform_best_time_to_remind_pred_output(output, output_proba):
    """
    Transform the prediction output for the best time to remind into a result dictionary.

    Parameters:
    - output (array): Model output array.
    - output_proba (array): Model output probabilities array.

    Returns:
    dict: Transformed result.
    """
    label_mapping = {0: 'Pagi', 1: 'Siang', 2: 'Sore', 3: 'Malam'}
    best_time_to_remind = label_mapping[output[0]]
    if best_time_to_remind == label_mapping[0]:
        reminder_context = (
            "Waktu yang tepat untuk mengirimkan pesan pengingat adalah pagi hari. Hal ini disebabkan oleh kesibukan yang mungkin dihadapi oleh para nasabah di waktu siang, sore, maupun malam."
        )
    elif best_time_to_remind == label_mapping[1]:
        reminder_context = (
            "Waktu yang tepat untuk mengirimkan pesan pengingat adalah siang hari. Hal ini disebabkan oleh kesibukan yang mungkin dihadapi oleh para nasabah di waktu pagi, sore, maupun malam."
        )
    elif best_time_to_remind == label_mapping[2]:
        reminder_context = (
            "Waktu yang tepat untuk mengirimkan pesan pengingat adalah sore hari. Hal ini disebabkan oleh kesibukan yang mungkin dihadapi oleh para nasabah di waktu pagi, siang, maupun sore."
        )
    elif best_time_to_remind == label_mapping[3]:
        reminder_context = (
            "Waktu yang tepat untuk mengirimkan pesan pengingat adalah malam hari. Hal ini disebabkan oleh kesibukan yang mungkin dihadapi oleh para nasabah di waktu pagi, siang, maupun sore."
        )
    else:
        reminder_context = "Tidak termasuk dalam pertimbangan kategori reschedule."
    best_time_to_remind_proba = rank_best_time_to_remind_output(output_proba, label_mapping)
    result = {
        'best_time_to_remind': best_time_to_remind,
        'reminder_context': reminder_context,
        'best_time_to_remind_probability': best_time_to_remind_proba
    }
    return result

def transform_best_time_to_follow_up_pred_output(output, output_proba):
    """
    Transform the prediction output for the best time to follow up into a result dictionary.

    Parameters:
    - output (array): Model output array.
    - output_proba (array): Model output probabilities array.

    Returns:
    dict: Transformed result.
    """
    label_mapping = {0: 'Pagi', 1: 'Siang', 2: 'Sore', 3: 'Malam'}
    best_time_to_follow_up = label_mapping[output[0]]
    best_time_to_follow_up_proba = rank_best_time_to_follow_up_output(output_proba, label_mapping)
    result = {
        'best_time_to_follow_up': best_time_to_follow_up,
        'best_time_to_follow_up_probability': best_time_to_follow_up_proba
    }
    return result

def rank_best_time_to_remind_output(result_proba, label_mapping):
    """
    Rank the best time to remind output probabilities.

    Parameters:
    - result_proba (array): Model output probabilities array.
    - label_mapping (dict): Mapping of labels to indices.

    Returns:
    list: Ranked list of best time to remind probabilities.
    """
    sorted_proba_indices = (-result_proba).argsort()
    rank_list = []
    for rank, idx in enumerate(sorted_proba_indices):
        label = label_mapping[idx]
        rank_list.append(f"{label} : {result_proba[idx]:.2%}")
    return rank_list

def rank_best_time_to_follow_up_output(result_proba, label_mapping):
    """
    Rank the best time to follow up output probabilities.

    Parameters:
    - result_proba (array): Model output probabilities array.
    - label_mapping (dict): Mapping of labels to indices.

    Returns:
    list: Ranked list of best time to follow up probabilities.
    """
    sorted_proba_indices = (-result_proba).argsort()
    rank_list = []
    for rank, idx in enumerate(sorted_proba_indices):
        label = label_mapping[idx]
        rank_list.append(f"{label} : {result_proba[idx]:.2%}")
    return rank_list

def rank_reschedule_eligibility_output(result_proba, label_mapping):
    """
    Rank the reschedule eligibility output probabilities.

    Parameters:
    - result_proba (array): Model output probabilities array.
    - label_mapping (dict): Mapping of labels to indices.

    Returns:
    list: Ranked list of reschedule eligibility probabilities.
    """
    sorted_proba_indices = (-result_proba).argsort()
    rank_list = []
    for rank, idx in enumerate(sorted_proba_indices):
        label = label_mapping[idx]
        rank_list.append(f"{label} : {result_proba[idx]:.2%}")
    return rank_list

class ObjectToCategoriesReminder(BaseEstimator, TransformerMixin):
    """
    Convert specified columns to categorical data type.

    Attributes:
    None
    """
    def fit(self, x, y=None):
        """
        Fit the encoder.

        Parameters:
            x (pd.DataFrame): Input DataFrame.
            y (pd.Series or None): Target variable (ignored).

        Returns:
            CategoricalEncoder: The fitted encoder.
        """
        return self

    def transform(self, x):
        x["debtor_working_time"] = x["debtor_working_time"].astype("category")
        x["debtor_previous_communication_channel"] = x["debtor_previous_communication_channel"].astype("category")
        x["last_interaction_type"] = x["last_interaction_type"].astype("category")
        x["reminder_response"] = x["reminder_response"].astype("category")
        return x

class ObjectToCategoriesFollowUp(BaseEstimator, TransformerMixin):
    """
    Convert specified columns to categorical data type.

    Attributes:
    None
    """
    def fit(self, x, y=None):
        """
        Fit the encoder.

        Parameters:
            x (pd.DataFrame): Input DataFrame.
            y (pd.Series or None): Target variable (ignored).

        Returns:
            CategoricalEncoder: The fitted encoder.
        """
        return self

    def transform(self, x):
        """
        Transform the input DataFrame by converting specified columns to categorical types.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with specified columns as categorical types.
        """
        x["debtor_aging"] = x["debtor_aging"].astype("category")
        x["debtor_working_time"] = x["debtor_working_time"].astype("category")
        x["debtor_previous_communication_channel"] = x["debtor_previous_communication_channel"].astype("category")
        x["debtor_field_communication"] = x["debtor_field_communication"].astype("category")
        x["last_interaction_type"] = x["last_interaction_type"].astype("category")
        x["follow_up_response"] = x["follow_up_response"].astype("category")
        return x
class DebtorWorkingTimeReminderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        """
        Fit the encoder.

        Parameters:
            x (pd.DataFrame): Input DataFrame.
            y (pd.Series or None): Target variable (ignored).

        Returns:
            CategoricalEncoder: The fitted encoder.
        """
        return self

    def transform(self, x):
        df_working_time = x[['debtor_working_time']]
        df_working_time = self.transform_debt_working_time(df_working_time)
        
        # Columns to check
        columns_to_check = ['busy_pagi', 'busy_siang', 'busy_sore', 'busy_malam']
        
        # Check if the columns already exist and apply transformation
        for column in columns_to_check:
            if column not in df_working_time.columns:
                df_working_time[column] = df_working_time.apply(getattr(self, f'_check_{column}'), axis=1)
        
        return df_working_time

    def transform_debt_working_time(self, df_working_time):
        """
        Transform debtor working time information into dummy variables.

        Parameters:
        - df_working_time (pd.DataFrame): DataFrame containing 'debtor_working_time' information.

        Returns:
        - pd.DataFrame: Transformed DataFrame with dummy variables.
        """
        debtor_working_time_dummy = []
        if df_working_time.values[0] == 'Malam-Pagi':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Pagi-Malam':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Pagi-Siang':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Pagi-Sore':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Siang-Malam':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Siang-Sore':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.values[0] == 'Siang-Sore':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        debtor_work_dummy = np.array(debtor_working_time_dummy).reshape((1, 7))
        debtor_work_df = pd.DataFrame(debtor_work_dummy, columns=['debtor_working_time_Malam-Pagi',
                              'debtor_working_time_Pagi-Malam', 
                              'debtor_working_time_Pagi-Siang', 
                              'debtor_working_time_Pagi-Sore', 
                              'debtor_working_time_Siang-Malam',
                              'debtor_working_time_Siang-Sore',
                              'debtor_working_time_Sore-Malam'])  
        return debtor_work_df
    def _check_busy_pagi(self, row):
        return 1 if (row['debtor_working_time_Pagi-Siang'] == 1) or \
                    (row['debtor_working_time_Pagi-Sore'] == 1) or \
                    (row['debtor_working_time_Pagi-Malam'] == 1) else 0

    def _check_busy_siang(self, row):
        return 1 if (row['debtor_working_time_Pagi-Siang'] == 1) or \
                    (row['debtor_working_time_Pagi-Sore'] == 1) or \
                    (row['debtor_working_time_Pagi-Malam'] == 1) or \
                    (row['debtor_working_time_Siang-Malam'] == 1) else 0

    def _check_busy_sore(self, row):
        return 1 if (row['debtor_working_time_Pagi-Malam'] == 1) or \
                    (row['debtor_working_time_Pagi-Sore'] == 1) or \
                    (row['debtor_working_time_Siang-Malam'] == 1) else 0

    def _check_busy_malam(self, row):
        return 1 if (row['debtor_working_time_Pagi-Malam'] == 1) or \
                    (row['debtor_working_time_Siang-Malam'] == 1) or \
                    (row['debtor_working_time_Malam-Pagi'] == 1) else 0

class DebtorWorkingTimeFollowUpTransformer(BaseEstimator, TransformerMixin):
    """
    Transform debtor working time information.
    """
    def transform_debt_working_time(self, df_working_time):
        """
        Transform debtor working time information.

        Parameters:
        - df_working_time (pd.DataFrame): Debtor working time DataFrame.

        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        debtor_working_time_dummy = []
        if df_working_time.iloc[0].iloc[0] == 'Malam-Pagi':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.iloc[0].iloc[0] == 'Pagi-Malam':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.iloc[0].iloc[0] == 'Pagi-Siang':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.iloc[0].iloc[0] == 'Pagi-Sore':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.iloc[0].iloc[0] == 'Siang-Malam':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.iloc[0].iloc[0] == 'Siang-Sore':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        if df_working_time.iloc[0].iloc[0] == 'Sore-Malam':
            debtor_working_time_dummy.append(1)
        else:
            debtor_working_time_dummy.append(0)
        debtor_work_dummy = np.array(debtor_working_time_dummy).reshape((1, 7))
        debtor_work_df = pd.DataFrame(debtor_work_dummy, columns=['debtor_working_time_Malam-Pagi',
                                                                  'debtor_working_time_Pagi-Malam',
                                                                  'debtor_working_time_Pagi-Siang',
                                                                  'debtor_working_time_Pagi-Sore',
                                                                  'debtor_working_time_Siang-Malam',
                                                                  'debtor_working_time_Siang-Sore',
                                                                  'debtor_working_time_Sore-Malam'])
        return debtor_work_df

    def _check_busy_pagi(self, row):
        """
        Check if debtor is busy during Pagi.

        Parameters:
        - row (pd.Series): A row from the DataFrame.

        Returns:
        int: 1 if busy, 0 otherwise.
        """
        return 1 if (
            row['debtor_working_time_Pagi-Siang'] == 1
            or row['debtor_working_time_Pagi-Sore'] == 1
            or row['debtor_working_time_Pagi-Malam'] == 1
        ) else 0

    def _check_busy_siang(self, row):
        """
        Check if debtor is busy during Siang.

        Parameters:
        - row (pd.Series): A row from the DataFrame.

        Returns:
        int: 1 if busy, 0 otherwise.
        """
        return 1 if (
            row['debtor_working_time_Pagi-Siang'] == 1
            or row['debtor_working_time_Pagi-Sore'] == 1
            or row['debtor_working_time_Pagi-Malam'] == 1
            or row['debtor_working_time_Siang-Malam'] == 1
        ) else 0

    def _check_busy_sore(self, row):
        """
        Check if debtor is busy during Sore.

        Parameters:
        - row (pd.Series): A row from the DataFrame.

        Returns:
        int: 1 if busy, 0 otherwise.
        """
        return 1 if (
            row['debtor_working_time_Pagi-Malam'] == 1
            or row['debtor_working_time_Pagi-Sore'] == 1
            or row['debtor_working_time_Siang-Malam'] == 1
        ) else 0

    def _check_busy_malam(self, row):
        """
        Check if debtor is busy during Malam.

        Parameters:
        - row (pd.Series): A row from the DataFrame.

        Returns:
        int: 1 if busy, 0 otherwise.
        """
        return 1 if (
            row['debtor_working_time_Pagi-Malam'] == 1
            or row['debtor_working_time_Siang-Malam'] == 1
            or row['debtor_working_time_Malam-Pagi'] == 1
        ) else 0


class FindBestTimeTarget(BaseEstimator, TransformerMixin):
    """
    Custom transformer to find the best time target based on busy time periods.

    Attributes:
    None
    """
    def fit(self, x, y=None):
        """
        Fit the encoder.

        Parameters:
            x (pd.DataFrame): Input DataFrame.
            y (pd.Series or None): Target variable (ignored).

        Returns:
            CategoricalEncoder: The fitted encoder.
        """
        return self

    def transform(self, x):
        """
        Transform the input DataFrame by adding a 'best_time_to_remind' column.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Transformed DataFrame with the 'best_time_to_remind' column.
        """
        x['best_time_to_remind'] = x.apply(self.find_best_time, axis=1)
        return x

    def find_best_time(self, row):
        """
        Determine the best time to remind based on the provided row information.

        Parameters:
        - row (pd.Series): A row from the DataFrame.

        Returns:
        str or None: The best time to remind or None if not applicable.
        """
        if (row['busy_pagi'] == 0) and (row['busy_siang'] == 0) and (row['busy_sore'] == 0) and (row['busy_malam'] == 0):
            return np.random.choice(["Pagi", "Siang", "Sore", "Malam"], p=[0.4, 0.4, 0.15, 0.05])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 0) and (row['busy_sore'] == 0) and (row['busy_malam'] == 1):
            return np.random.choice(["Pagi", "Siang", "Sore"], p=[0.4, 0.4, 0.2])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 0) and (row['busy_sore'] == 1) and (row['busy_malam'] == 0):
            return np.random.choice(["Pagi", "Siang", "Malam"], p=[0.4, 0.5, 0.1])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 1) and (row['busy_sore'] == 0) and (row['busy_malam'] == 0):
            return np.random.choice(["Pagi", "Sore", "Malam"], p=[0.5, 0.4, 0.1])
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 0) and (row['busy_sore'] == 0) and (row['busy_malam'] == 0):
            return np.random.choice(["Siang", "Sore", "Malam"], p=[0.6, 0.3, 0.1])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 0) and (row['busy_sore'] == 1) and (row['busy_malam'] == 1):
            return np.random.choice(["Pagi", "Siang"], p=[0.4, 0.6])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 1) and (row['busy_sore'] == 0) and (row['busy_malam'] == 1):
            return np.random.choice(["Pagi", "Sore"], p=[0.55, 0.45])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 1) and (row['busy_sore'] == 1) and (row['busy_malam'] == 0):
            return np.random.choice(["Pagi", "Malam"], p=[0.7, 0.3])
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 1) and (row['busy_sore'] == 0) and (row['busy_malam'] == 0):
            return np.random.choice(["Sore", "Malam"], p=[0.55, 0.45])
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 0) and (row['busy_sore'] == 1) and (row['busy_malam'] == 0):
            return np.random.choice(["Siang", "Malam"], p=[0.8, 0.2])
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 0) and (row['busy_sore'] == 0) and (row['busy_malam'] == 1):
            return np.random.choice(["Siang", "Sore"], p=[0.8, 0.2])
        elif (row['busy_pagi'] == 0) and (row['busy_siang'] == 1) and (row['busy_sore'] == 1) and (row['busy_malam'] == 1):
            return "Pagi"
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 0) and (row['busy_sore'] == 1) and (row['busy_malam'] == 1):
            return "Siang"
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 1) and (row['busy_sore'] == 0) and (row['busy_malam'] == 1):
            return "Sore"
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 1) and (row['busy_sore'] == 1) and (row['busy_malam'] == 0):
            return "Malam"
        elif (row['busy_pagi'] == 1) and (row['busy_siang'] == 1) and (row['busy_sore'] == 1) and (row['busy_malam'] == 1):
            return np.random.choice(["Siang", "Malam"], p=[0.6, 0.4])
        else:
            return None

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features based on predefined mappings.

    Attributes:
        mapping (dict): A dictionary containing mappings for categorical features.

    Methods:
        fit(X, y=None): Fit the encoder (no action needed).
        transform(X): Transform the input DataFrame by applying the mappings.
    """

    def __init__(self):
        self.mapping = {
            "best_time_to_remind": {"Pagi": 0, "Siang": 1, "Sore": 2, "Malam": 3},
        }

    def fit(self, x, y=None):
        """
        Fit the encoder.

        Parameters:
            x (pd.DataFrame): Input DataFrame.
            y (pd.Series or None): Target variable (ignored).

        Returns:
            CategoricalEncoder: The fitted encoder.
        """
        return self

    def transform(self, x):
        """
        Transform the input DataFrame by applying the mappings.

        Parameters:
            x (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        x_encoded = x.copy()
        for column, mapping in self.mapping.items():
            if (
                column in x_encoded.columns
                and x_encoded[column].dtype == 'object'
            ):
                x_encoded[column] = x_encoded[column].map(mapping)
        return x_encoded

data_preprocessing_best_time_reminder = Pipeline([
    ('object_to_categories', ObjectToCategoriesReminder()),
    ('debtor_working_time_reminder_transformer', DebtorWorkingTimeReminderTransformer()),
])

data_preprocessing_best_time_follow_up = Pipeline([
    ('object_to_categories', ObjectToCategoriesFollowUp()),
    ('debtor_working_time_follow_up_transformer', DebtorWorkingTimeFollowUpTransformer()),
])