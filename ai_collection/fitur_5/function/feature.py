"""
Module for inference, data transformation, and preprocessing.

This module contains methods for making predictions using machine learning models,
as well as data transformation and preprocessing.

Methods:
    - predict_best_time_to_bill: Predicts the best time to bill for a given debtor 
    and updates the dataset.
    - predict_recommended_collectors_assignments: Predicts recommended collectors 
    assignments for a given debtor and updates the dataset.
    - predict_interaction_efficiency: Predicts interaction efficiency for a given 
    debtor and updates the dataset.
    - transform_input_debtor: Transforms input data for debtor predictions into a DataFrame.
    - transform_best_time_to_bill_pred_output: Transforms the output of best time to bill 
    prediction.
    - transform_recommended_collectors_output: Transforms the output of recommended collectors 
    prediction.
    - transform_interaction_eficiency_output: Transforms the output of interaction efficiency 
    prediction.
    - rank_best_time_to_bill_output: Ranks the best time to bill prediction probabilities.
    - ObjectToCategoriesDebtor: Transformer class for converting object columns to categorical 
    for debtor data.
    - ObjectToCategoriesCollector: Transformer class for converting object columns to categorical 
    for collector data.

"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..apps import Fitur5Config
from ..libraries import utils


def predict_best_time_to_bill(data):
    """
    Predicts the best time to bill for a given debtor and updates the dataset.

    Args:
        data (dict): Input data for predicting the best time to bill.

    Returns:
        dict: Result of the prediction.
    """
    model = Fitur5Config.besttime_to_bill_model
    dataset_file_name = 'debtor_v05_231005.csv'
    input_df = transform_input_debtor(data)
    preprocessed_df = data_preprocessing_best_time.fit_transform(input_df)
    output = model.predict(preprocessed_df)
    output_proba = model.predict_proba(preprocessed_df).flatten()
    result = transform_best_time_to_bill_pred_output(output, output_proba)
    utils.append_dataset_with_new_data(dataset_file_name, input_df, result)
    return result

def predict_recommended_collectors_assignments(data):
    """
    Predicts recommended collectors assignments for a given debtor and updates the dataset.

    Args:
        data (dict): Input data for predicting recommended collectors assignments.

    Returns:
        dict: Result of the prediction.
    """
    model = Fitur5Config.recsys_collector_assignments_model
    dataset_file_name_coll = 'collector_v03_231005.csv'

    input_debtor_df = transform_input_debtor(data)
    dataset_coll_path = utils.load_dataset_path(dataset_file_name_coll)
    dt_collector = pd.read_csv(dataset_coll_path)

    preprocessed_debtor = data_preprocessing_debtor.fit_transform(input_debtor_df)
    preprocessed_collector = data_preprocessing_collector.fit_transform(dt_collector)
    xdebt, xcoll = scale_for_recsys(preprocessed_debtor, preprocessed_collector)

    output = model.predict([xdebt, xcoll])
    result = transform_recommended_collectors_output(output, input_debtor_df, dt_collector)
    return result

def predict_interaction_efficiency(data):
    """
    Predicts interaction efficiency for a given debtor and updates the dataset.

    Args:
        data (dict): Input data for predicting interaction efficiency.

    Returns:
        dict: Result of the prediction.
    """
    model = Fitur5Config.interaction_eficiency_model
    dataset_file_name_inter = 'collector_performances_v03_231005.csv'
    input_df = transform_input_debtor(data)
    dataset_inter_path = utils.load_dataset_path(dataset_file_name_inter)
    interactions_df = pd.read_csv(dataset_inter_path)
    preprocessed_input = (
        data_preprocessing_input_inter.fit_transform(input_df)
        .reset_index(drop=True)
    )

    preprocessed_inter = (
        data_preprocessing_interactions.fit_transform(interactions_df)
        .reset_index(drop=True)
    )
    preprocessed_input = preprocessed_input.reset_index(drop=True)
    preprocessed_inter = preprocessed_inter.reset_index(drop=True)
    combined_df = pd.concat([preprocessed_inter, preprocessed_input], ignore_index=True)
    pca_df = scale_and_pca.fit_transform(combined_df)
    y_clust = model.predict(pca_df)
    combined_df["clusters"] = y_clust
    result = transform_interaction_eficiency_output(combined_df)
    return result

def transform_input_debtor(data):
    """
    Transforms input data for debtor predictions into a DataFrame.

    Args:
        data (dict): Input data for debtor predictions.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
    return df


def transform_best_time_to_bill_pred_output(output, output_proba):
    """
    Transforms the output of best time to bill prediction.

    Args:
        output (array-like): Model output.
        output_proba (array-like): Model output probabilities.

    Returns:
        dict: Transformed result.
    """
    label_mapping = {0: 'Pagi', 1: 'Siang', 2: 'Sore', 3: 'Malam'}
    best_time_to_bill = label_mapping[output[0]]
    best_time_to_bill_proba = rank_best_time_to_bill_output(output_proba, label_mapping)
    result = {
        'best_time_to_bill': best_time_to_bill,
        'best_time_to_bill_probability': best_time_to_bill_proba
    }
    return result


def transform_recommended_collectors_output(output, input_debtor_df, dt_collector):
    """
    Transforms the output of recommended collectors prediction.

    Args:
        output (array-like): Model output.
        input_debtor_df (pd.DataFrame): Input DataFrame for debtor data.
        dt_collector (pd.DataFrame): DataFrame for collector data.

    Returns:
        dict: Transformed result.
    """
    output = pd.DataFrame(output)
    output["distance"] = calculate_distance_per_collector(input_debtor_df, dt_collector)
    output.sort_values(by=[0, 'distance'], ascending=[False, True], inplace=True)
    recommended_collectors_index = output[:10].index
    recommended_collectors = (
        dt_collector
        .loc[recommended_collectors_index, 'collector_name']
        .tolist()
    )

    result = {
        'recommended_collectors_to_assign': recommended_collectors
    }
    return result


def transform_interaction_eficiency_output(combined_df):
    """
    Transforms the output of interaction efficiency prediction.

    Args:
        combined_df (pd.DataFrame): Combined DataFrame from interaction efficiency prediction.

    Returns:
        dict: Transformed result.
    """
    # takes the output clusters on input data
    cluster = combined_df['clusters'].iloc[-1]
    if cluster == 0:
        cat_cluster = "Efisien dalam Interaksi dengan Pelanggan"
    elif cluster == 1:
        cat_cluster = "Efisien dalam Mobilitas/Perjalanan"
    elif cluster == 2:
        cat_cluster = "Efisien dalam Manajemen Waktu"
    elif cluster == 3:
        cat_cluster = "Efisien dalam Proses Aktifitas"
    elif cluster == 4:
        cat_cluster = "Efisien dalam Respon dan Interaksi"
    result = {
        'category_cluster': cat_cluster
    }
    return result

def rank_best_time_to_bill_output(result_proba, label_mapping):
    """
    Ranks the best time to bill prediction probabilities.

    Args:
        result_proba (array-like): Model output probabilities.
        label_mapping (dict): Mapping of labels to categories.

    Returns:
        list: Ranks of categories with probabilities.
    """
    sorted_proba_indices = (-result_proba).argsort()
    rank_list = []
    for rank, idx in enumerate(sorted_proba_indices):
        label = label_mapping[idx]
        rank_list.append(f"{label} : {result_proba[idx]:.2%}")
    return rank_list

class ObjectToCategoriesDebtor(BaseEstimator, TransformerMixin):
    """
    Transformer class for converting object columns to categorical for debtor data.
    """

    def fit(self, x, y=None):
        """
        Fit method conforming to the sklearn transformer interface.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.
        - y: Unused parameter.

        Returns:
        self
        """
        return self

    def transform(self, x):
        """
        Transform method to convert specified object columns to categorical.

        Parameters:
        - X (pd.DataFrame): Input DataFrame with debtor data.

        Returns:
        pd.DataFrame: Transformed DataFrame with specified object columns converted to categorical.
        """
        categorical_columns = ['debtor_gender', 'debtor_education_level',
                               'employment_status', 'debtor_working_day',
                               'debtor_working_time']

        x[categorical_columns] = x[categorical_columns].astype('category')
        return x


class ObjectToCategoriesCollector(BaseEstimator, TransformerMixin):
    """
    Transformer class for converting object columns to categorical for collector data.
    """

    def fit(self, x, y=None):
        """
        Fit method conforming to the sklearn transformer interface.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.
        - y: Unused parameter.

        Returns:
        self
        """
        return self

    def transform(self, x):
        """
        Transform method to convert specified object columns to categorical.

        Parameters:
        - X (pd.DataFrame): Input DataFrame with collector data.

        Returns:
        pd.DataFrame: Transformed DataFrame with specified object columns converted to categorical.
        """
        categorical_columns=['collector_education_level','collector_workday','collector_worktime']

        x[categorical_columns] = x[categorical_columns].astype('category')
        return x

class DebtorWorkingTimeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class for working time-related features of debtors.

    Methods:
    - fit: Fit method required by scikit-learn transformer interface.
    - transform: Transform method required by scikit-learn transformer interface.
    - transform_debt_working_time: Transform method specifically for debtor working time.
    - _check_busy_pagi: Check if debtor is busy during morning time.
    - _check_busy_siang: Check if debtor is busy during afternoon time.
    - _check_busy_sore: Check if debtor is busy during evening time.
    - _check_busy_malam: Check if debtor is busy during night time.

    """
    def fit(self, x, y=None):
        """
        Fit method required by scikit-learn transformer interface.

        Args:
            x (pd.DataFrame): Input data.
            y (array-like): Target values.

        Returns:
            self: Returns an instance of self.
        """
        return self
    def transform(self, x):
        """
        Transform method required by scikit-learn transformer interface.

        Args:
            x (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data.
        """
        debt_working_time = x[['debtor_working_time']]
        debt_working_time = self.transform_debt_working_time(debt_working_time)
        # Check if the columns already exist
        if 'busy_pagi' not in debt_working_time.columns:
            debt_working_time['busy_pagi'] = debt_working_time.apply(
                self._check_busy_pagi,
                axis=1
            )
        if 'busy_siang' not in debt_working_time.columns:
            debt_working_time['busy_siang'] = debt_working_time.apply(
                self._check_busy_siang,
                axis=1
            )
        if 'busy_sore' not in debt_working_time.columns:
            debt_working_time['busy_sore'] = debt_working_time.apply(
               self._check_busy_sore,
               axis=1
            )
        if 'busy_malam' not in debt_working_time.columns:
            debt_working_time['busy_malam'] = debt_working_time.apply(
               self._check_busy_malam,
               axis=1
            )
        return debt_working_time

    def transform_debt_working_time(self, debt_working_time):
        """
        Transform method specifically for debtor working time.

        Args:
            debt_working_time (pd.DataFrame): Data related to debtor working time.

        Returns:
            pd.DataFrame: Transformed data.
        """
        debt_working_time_dum = []
        if debt_working_time.values[0] == 'Malam-Pagi':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Pagi-Malam':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Pagi-Siang':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Pagi-Sore':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Siang-Malam':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Siang-Sore':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Sore-Malam':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)

        debt_work_dum = np.array(debt_working_time_dum).reshape((1, 7))

        debt_work_df = pd.DataFrame(debt_work_dum, columns=['debtor_working_time_Malam-Pagi',
                              'debtor_working_time_Pagi-Malam', 
                              'debtor_working_time_Pagi-Siang', 
                              'debtor_working_time_Pagi-Sore', 
                              'debtor_working_time_Siang-Malam',
                              'debtor_working_time_Siang-Sore',
                              'debtor_working_time_Sore-Malam'])  
        return debt_work_df

    def _check_busy_pagi(self, row):
        """
        Check if debtor is busy during morning time.

        Args:
            row (pd.Series): Row of data.

        Returns:
            int: 1 if busy, else 0.
        """
        return 1 if (
            (row['debtor_working_time_Pagi-Siang'] == 1) or
            (row['debtor_working_time_Pagi-Sore'] == 1) or
            (row['debtor_working_time_Pagi-Malam'] == 1)
        ) else 0

    def _check_busy_siang(self, row):
        """
        Check if debtor is busy during afternoon time.

        Args:
            row (pd.Series): Row of data.

        Returns:
            int: 1 if busy, else 0.
        """
        return 1 if (
            (row['debtor_working_time_Pagi-Siang'] == 1) or
            (row['debtor_working_time_Pagi-Sore'] == 1) or
            (row['debtor_working_time_Pagi-Malam'] == 1) or
            (row['debtor_working_time_Siang-Malam'] == 1)
        ) else 0

    def _check_busy_sore(self, row):
        """
        Check if debtor is busy during evening time.

        Args:
            row (pd.Series): Row of data.

        Returns:
            int: 1 if busy, else 0.
        """
        return 1 if (
            (row['debtor_working_time_Pagi-Malam'] == 1) or
            (row['debtor_working_time_Pagi-Sore'] == 1) or
            (row['debtor_working_time_Siang-Malam'] == 1)
        ) else 0

    def _check_busy_malam(self, row):
        """
        Check if debtor is busy during night time.

        Args:
            row (pd.Series): Row of data.

        Returns:
            int: 1 if busy, else 0.
        """
        return 1 if (
            (row['debtor_working_time_Pagi-Malam'] == 1) or
            (row['debtor_working_time_Siang-Malam'] == 1) or
            (row['debtor_working_time_Malam-Pagi'] == 1)
        ) else 0


class FindBestTimeTarget(BaseEstimator, TransformerMixin):
    """
    Transformer class for finding the best time to bill for debtors.

    Methods:
    - fit: Fit method required by scikit-learn transformer interface.
    - transform: Transform method required by scikit-learn transformer interface.
    - find_best_time: Find the best time to bill based on the debtor's availability.

    """

    def fit(self, x, y=None):
        """
        Fit method required by scikit-learn transformer interface.

        Args:
            x (pd.DataFrame): Input data.
            y (array-like): Target values.

        Returns:
            self: Returns an instance of self.
        """
        return self

    def transform(self, x):
        """
        Transform method required by scikit-learn transformer interface.

        Args:
            x (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data.
        """
        x['best_time_to_bill'] = x.apply(self.find_best_time, axis=1)
        return x

    def find_best_time(self, row):
        """
        Find the best time to bill based on the debtor's availability.

        Args:
            row (pd.Series): Row of data.

        Returns:
            str or None: Best time to bill or None if not determined.
        """
        busy_times = [
            'busy_pagi', 'busy_siang', 'busy_sore', 'busy_malam'
        ]

        combinations = [
            ['Pagi', 'Siang', 'Sore', 'Malam'],
            ['Pagi', 'Siang', 'Sore'],
            ['Pagi', 'Siang', 'Malam'],
            ['Pagi', 'Sore', 'Malam'],
            ['Siang', 'Sore', 'Malam'],
            ['Pagi', 'Siang'],
            ['Pagi', 'Sore'],
            ['Pagi', 'Malam'],
            ['Sore', 'Malam'],
            ['Siang', 'Malam'],
            ['Siang', 'Sore'],
            ['Pagi'],
            ['Siang'],
            ['Sore'],
            ['Malam'],
            ['Siang', 'Malam']
        ]

        for i, combination in enumerate(combinations):
            if all(row[busy] == 0 for busy in busy_times[i]):
                return np.random.choice(combination, p=self._get_probability(i))

        return None

    def _get_probability(self, index):
        probabilities = [
            [0.4, 0.4, 0.15, 0.05],
            [0.4, 0.4, 0.2],
            [0.4, 0.5, 0.1],
            [0.5, 0.4, 0.1],
            [0.6, 0.3, 0.1],
            [0.4, 0.6],
            [0.55, 0.45],
            [0.7, 0.3],
            [0.55, 0.45],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.6, 0.4]
        ]
        return probabilities[index]


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer for encoding categorical features.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    mapping : dict
        A dictionary containing mappings for each categorical feature to be encoded.

    Methods
    -------
    fit(x, y=None):
        Fit the encoder. This method does nothing in this implementation 
        as there is no training involved.

        Parameters:
        -----------
        x : DataFrame, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,), optional (default=None)
            The target labels.

        Returns:
        --------
        self : object
            Returns self.

    transform(x):
        Transform the input data by encoding categorical features based on pre-defined mappings.

        Parameters:
        -----------
        x : DataFrame, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        X_encoded : DataFrame, shape (n_samples, n_features)
            The transformed data with encoded categorical features.

    Example:
    --------
    # Instantiate the CategoricalEncoder
    encoder = CategoricalEncoder()

    # Fit and transform the input data
    X_encoded = encoder.fit_transform(X)
    """
    def __init__(self):
        """
        Initialize the CategoricalEncoder.

        Parameters:
        -----------
        None
        """
        self.mapping = {
            "best_time_to_bill": {"Pagi" : 0, 
                                  "Siang": 1,
                                  "Sore" : 2,
                                  "Malam": 3},
            "transportation_type": {"Motor" : 0,
                                    "Mobil": 1}
        }

    def fit(self, x, y=None):
        """
        Fit the encoder. This method does nothing in this implementation 
        as there is no training involved.

        Parameters:
        -----------
        x : DataFrame, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,), optional (default=None)
            The target labels.

        Returns:
        --------
        self : object
            Returns self.
        """
        return self

    def transform(self, x):
        """
        Transform the input data by encoding categorical features based on pre-defined mappings.

        Parameters:
        -----------
        x : DataFrame, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        X_encoded : DataFrame, shape (n_samples, n_features)
            The transformed data with encoded categorical features.
        """
        x_encoded = x.copy()
        for column, mapping in self.mapping.items():
            if column in x_encoded.columns and x_encoded[column].dtype == 'object':
                x_encoded[column] = x_encoded[column].map(mapping)
        return x_encoded


class PreprocessDebtorForRecSys(BaseEstimator, TransformerMixin):
    """
    PreprocessDebtorForRecSys class extends BaseEstimator and TransformerMixin
    for preprocessing debtor data for a recommendation system.
    """
    def fit(self, x, y=None):
        """
        Fit the transformer. This method does nothing in this implementation 
        as there is no training involved.

        Parameters:
        -----------
        x : DataFrame, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,), optional (default=None)
            The target labels.

        Returns:
        --------
        self : object
            Returns self.
        """
        return self

    def transform(self, x):
        """
        Transform the input data by preprocessing debtor features.

        Parameters:
        -----------
        x : DataFrame, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        dt_debtor : DataFrame, shape (n_samples, n_features)
            The preprocessed debtor data.
        """
        dt_debt = x[['debtor_age', 'debtor_latitude',
                       'debtor_longitude']]

        debt_edulvl = self._transform_debt_edu_level(x['debtor_education_level'])
        debt_workday = self._transform_debt_working_day(x['debtor_working_day'])
        debt_worktime = self._transform_debt_working_time(x['debtor_working_time'])
        dt_debtor = pd.concat([dt_debt, debt_edulvl, debt_workday, debt_worktime], axis=1)
        dt_debtor = self._generate_luang_debtor(dt_debtor)
        dt_debtor = pd.concat([dt_debtor] * 1000)

        return dt_debtor.reset_index(drop=True)

    def _transform_debt_edu_level(self, debt_edu_level):
        """
        Transform debtor education level into dummy variables.

        Parameters:
        -----------
        debt_edu_level : Series
            Debtor education level data.

        Returns:
        --------
        debt_edu_df : DataFrame
            Transformed debtor education level data.
        """
        edu_levels = ['D3', 'D4', 'S1', 'S2', 'S3', 'SD', 'SMA', 'SMP']

        debt_edu_level_dum = [1 if debt_edu_level.values[0] == level else 0 for level in edu_levels]

        debt_edu_dum = np.array(debt_edu_level_dum).reshape((1, len(edu_levels)))

        columns = [f'debtor_education_level_{level}' for level in edu_levels]
        debt_edu_df = pd.DataFrame(debt_edu_dum, columns=columns)

        return debt_edu_df


    def _transform_debt_working_day(self, debt_working_day):
        """
        Transform debtor working day into dummy variables.

        Parameters:
        -----------
        debt_working_day : Series
            Debtor working day data.

        Returns:
        --------
        debt_work_df : DataFrame
            Transformed debtor working day data.
        """
        debt_working_day_dum = []
        if debt_working_day.values[0] == 'Sabtu-Minggu':
            debt_working_day_dum.append(1)
        else:
            debt_working_day_dum.append(0)
        if debt_working_day.values[0] == 'Senin-Jumat':
            debt_working_day_dum.append(1)
        else:
            debt_working_day_dum.append(0)
        if debt_working_day.values[0] == 'Senin-Minggu':
            debt_working_day_dum.append(1)
        else:
            debt_working_day_dum.append(0)


        debt_work_dum = np.array(debt_working_day_dum).reshape((1, 3))

        debt_work_df = pd.DataFrame(debt_work_dum, columns=['debtor_workday_Sabtu-Minggu',
                              'debtor_workday_Senin-Jumat',
                              'debtor_workday_Senin-Minggu'])

        return debt_work_df

    def _transform_debt_working_time(self, debt_working_time):
        """
        Transform the 'debt_working_time' data into a one-hot encoded DataFrame 
        representing different working time categories.

        Parameters:
        - debt_working_time (pd.Series): A pandas Series containing working time information.

        Returns:
        pd.DataFrame: A one-hot encoded DataFrame with columns representing 
        different working time categories.
        """
        debt_working_time_dum = []
        if debt_working_time.values[0] == 'Malam-Pagi':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Pagi-Malam':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Pagi-Siang':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Pagi-Sore':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Siang-Malam':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Siang-Sore':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)
        if debt_working_time.values[0] == 'Sore-Malam':
            debt_working_time_dum.append(1)
        else:
            debt_working_time_dum.append(0)

        debt_work_dum = np.array(debt_working_time_dum).reshape((1, 7))

        debt_work_df = pd.DataFrame(debt_work_dum, columns=['debtor_worktime_Malam-Pagi',
                              'debtor_worktime_Pagi-Malam',
                              'debtor_worktime_Pagi-Siang',
                              'debtor_worktime_Pagi-Sore',
                              'debtor_worktime_Siang-Malam',
                              'debtor_worktime_Siang-Sore',
                              'debtor_worktime_Sore-Malam'])

        return debt_work_df

    def _generate_luang_debtor(self, dt_debtor):
        """
        Generate additional columns in the 'dt_debtor' DataFrame to represent 
        various time slots for leisure activities.

        Parameters:
        - dt_debtor (pd.DataFrame): Input DataFrame containing debtor information.

        Returns:
        pd.DataFrame: Modified DataFrame with additional columns representing leisure time slots.
        """
        # Check if the columns already exist
        if 'debtor_luang_pagi' not in dt_debtor.columns:
            dt_debtor['debtor_luang_pagi'] = dt_debtor.apply(
                self._check_debtor_luang_pagi, axis=1
            )
        if 'debtor_luang_siang' not in dt_debtor.columns:
            dt_debtor['debtor_luang_siang'] = dt_debtor.apply(
                self._check_debtor_luang_siang, axis=1
            )
        if 'debtor_luang_sore' not in dt_debtor.columns:
            dt_debtor['debtor_luang_sore'] = dt_debtor.apply(
                self._check_debtor_luang_sore, axis=1
            )
        if 'debtor_luang_malam' not in dt_debtor.columns:
            dt_debtor['debtor_luang_malam'] = dt_debtor.apply(
                self._check_debtor_luang_malam, axis=1
            )
        if 'debtor_luang_senin' not in dt_debtor.columns:
            dt_debtor['debtor_luang_senin'] = dt_debtor.apply(
                self._check_debtor_luang_senin, axis=1
            )
        if 'debtor_luang_selasa' not in dt_debtor.columns:
            dt_debtor['debtor_luang_selasa'] = dt_debtor.apply(
                self._check_debtor_luang_selasa, axis=1
            )
        if 'debtor_luang_rabu' not in dt_debtor.columns:
            dt_debtor['debtor_luang_rabu'] = dt_debtor.apply(
                self._check_debtor_luang_rabu, axis=1
            )
        if 'debtor_luang_kamis' not in dt_debtor.columns:
            dt_debtor['debtor_luang_kamis'] = dt_debtor.apply(
                self._check_debtor_luang_kamis, axis=1
            )
        if 'debtor_luang_jumat' not in dt_debtor.columns:
            dt_debtor['debtor_luang_jumat'] = dt_debtor.apply(
                self._check_debtor_luang_jumat, axis=1
            )
        if 'debtor_luang_sabtu' not in dt_debtor.columns:
            dt_debtor['debtor_luang_sabtu'] = dt_debtor.apply(
                self._check_debtor_luang_sabtu, axis=1
            )
        if 'debtor_luang_minggu' not in dt_debtor.columns:
            dt_debtor['debtor_luang_minggu'] = dt_debtor.apply(
                self._check_debtor_luang_minggu, axis=1
            )

        dt_debtor.drop(['debtor_workday_Sabtu-Minggu','debtor_workday_Senin-Jumat',
                   'debtor_workday_Senin-Minggu', 'debtor_worktime_Malam-Pagi',
                   'debtor_worktime_Pagi-Malam', 'debtor_worktime_Pagi-Siang',
                   'debtor_worktime_Pagi-Sore', 'debtor_worktime_Siang-Malam',
                   'debtor_worktime_Siang-Sore', 'debtor_worktime_Sore-Malam'], 
                   axis=1, inplace=True)
        return dt_debtor

    def _check_debtor_luang_pagi(self, row):
        """
        Check if the debtor has leisure time in the morning (Pagi).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the morning, 0 otherwise.
        """
        return 0 if (
            (row['debtor_worktime_Pagi-Siang'] == 1) or
            (row['debtor_worktime_Pagi-Sore'] == 1) or
            (row['debtor_worktime_Pagi-Malam'] == 1)
        ) else 1

    def _check_debtor_luang_siang(self, row):
        """
        Check if the debtor has leisure time in the afternoon (Siang).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the afternoon, 0 otherwise.
        """
        return 0 if (
            (row['debtor_worktime_Pagi-Siang'] == 1) or
            (row['debtor_worktime_Pagi-Sore'] == 1) or
            (row['debtor_worktime_Pagi-Malam'] == 1) or
            (row['debtor_worktime_Siang-Malam'] == 1) or
            (row['debtor_worktime_Siang-Sore'] == 1)
        ) else 1

    def _check_debtor_luang_sore(self, row):
        """
        Check if the debtor has leisure time in the evening (Sore).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the evening, 0 otherwise.
        """
        return 0 if (
            (row['debtor_worktime_Pagi-Malam'] == 1) or
            (row['debtor_worktime_Pagi-Sore'] == 1) or
            (row['debtor_worktime_Siang-Malam'] == 1) or
            (row['debtor_worktime_Siang-Sore'] == 1) or
            (row['debtor_worktime_Sore-Malam'] == 1)
        ) else 1

    def _check_debtor_luang_malam(self, row):
        """
        Check if the debtor has leisure time in the night (Malam).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the night, 0 otherwise.
        """
        return 0 if (
            (row['debtor_worktime_Pagi-Malam'] == 1) or
            (row['debtor_worktime_Siang-Malam'] == 1) or
            (row['debtor_worktime_Malam-Pagi'] == 1) or
            (row['debtor_worktime_Sore-Malam'] == 1)
        ) else 1

    def _check_debtor_luang_senin(self, row):
        """
        Check if the debtor has leisure time in the Monday (Senin).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the Monday, 0 otherwise.
        """
        return 0 if (
            (row['debtor_workday_Senin-Jumat'] == 1) or
            (row['debtor_workday_Senin-Minggu'] == 1)
        ) else 1

    def _check_debtor_luang_selasa(self, row):
        """
        Check if the debtor has leisure time in the Tuesday (Selasa).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the Tuesday, 0 otherwise.
        """
        return 0 if (
            (row['debtor_workday_Senin-Jumat'] == 1) or
            (row['debtor_workday_Senin-Minggu'] == 1)
        ) else 1

    def _check_debtor_luang_rabu(self, row):
        """
        Check if the debtor has leisure time in the Wednesday (Rabu).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the Wednesday, 0 otherwise.
        """
        return 0 if (
            (row['debtor_workday_Senin-Jumat'] == 1) or
            (row['debtor_workday_Senin-Minggu'] == 1)
        ) else 1

    def _check_debtor_luang_kamis(self, row):
        """
        Check if the debtor has leisure time in the Thursday (Kamis).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the Thursday, 0 otherwise.
        """
        return 0 if (
            (row['debtor_workday_Senin-Jumat'] == 1) or
            (row['debtor_workday_Senin-Minggu'] == 1)
        ) else 1

    def _check_debtor_luang_jumat(self, row):
        """
        Check if the debtor has leisure time in the Friday (Jumat).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the Friday, 0 otherwise.
        """
        return 0 if (
            (row['debtor_workday_Senin-Jumat'] == 1) or
            (row['debtor_workday_Senin-Minggu'] == 1)
        ) else 1

    def _check_debtor_luang_sabtu(self, row):
        """
        Check if the debtor has leisure time in the Saturday (Sabtu).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the Saturday, 0 otherwise.
        """
        return 0 if (
            (row['debtor_workday_Sabtu-Minggu'] == 1) or
            (row['debtor_workday_Senin-Minggu'] == 1)
        ) else 1

    def _check_debtor_luang_minggu(self, row):
        """
        Check if the debtor has leisure time in the Sunday (Minggu).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of debtor information.

        Returns:
        int: 1 if the debtor has leisure time in the Sunday, 0 otherwise.
        """
        return 0 if (
            (row['debtor_workday_Sabtu-Minggu'] == 1) or
            (row['debtor_workday_Senin-Minggu'] == 1)
        ) else 1


class PreprocessCollectorForRecSys(BaseEstimator, TransformerMixin):
    """
    PreprocessCollectorForRecSys class extends BaseEstimator and TransformerMixin
    for preprocessing collector data for a recommendation system.
    """
    def fit(self, x, y=None):
        """
        Fit method to conform to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.
        - y: Unused parameter.

        Returns:
        self
        """
        return self

    def transform(self, x):
        """
        Transform method to preprocess collector data.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Processed DataFrame.
        """
        dt_collector = x[['collector_age','collector_latitude', 'collector_longitude',
                          'collector_education_level','collector_workday','collector_worktime']]

        dt_collector = pd.get_dummies(dt_collector, columns=['collector_education_level',
                                                      'collector_workday','collector_worktime'])
        dt_collector = self._generate_kerja_collector(dt_collector)
        dt_collector = self._adjust_education_level(dt_collector)

        return dt_collector.reset_index(drop=True)

    def _generate_kerja_collector(self, dt_collector):
        """
        Generate additional columns in 'dt_collector' DataFrame to represent 
        various working time slots.

        Parameters:
        - dt_collector (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Modified DataFrame with additional working time columns.
        """
        # Check if the columns already exist
        if 'collector_kerja_pagi' not in dt_collector.columns:
            dt_collector['collector_kerja_pagi'] = dt_collector.apply(
                self._check_collector_kerja_pagi,
                axis=1
            )
        if 'collector_kerja_siang' not in dt_collector.columns:
            dt_collector['collector_kerja_siang'] = dt_collector.apply(
                self._check_collector_kerja_siang,
                axis=1
            )
        if 'collector_kerja_sore' not in dt_collector.columns:
            dt_collector['collector_kerja_sore'] = dt_collector.apply(
                self._check_collector_kerja_sore,
                axis=1
            )
        if 'collector_kerja_malam' not in dt_collector.columns:
            dt_collector['collector_kerja_malam'] = dt_collector.apply(
                self._check_collector_kerja_malam,
                axis=1
            )
        if 'collector_kerja_senin' not in dt_collector.columns:
            dt_collector['collector_kerja_senin'] = dt_collector.apply(
                self._check_collector_kerja_senin,
                axis=1
            )
        if 'collector_kerja_selasa' not in dt_collector.columns:
            dt_collector['collector_kerja_selasa'] = dt_collector.apply(
                self._check_collector_kerja_selasa,
                axis=1
            )
        if 'collector_kerja_rabu' not in dt_collector.columns:
            dt_collector['collector_kerja_rabu'] = dt_collector.apply(
                self._check_collector_kerja_rabu,
                axis=1
            )
        if 'collector_kerja_kamis' not in dt_collector.columns:
            dt_collector['collector_kerja_kamis'] = dt_collector.apply(
                self._check_collector_kerja_kamis,
                axis=1
            )
        if 'collector_kerja_jumat' not in dt_collector.columns:
            dt_collector['collector_kerja_jumat'] = dt_collector.apply(
                self._check_collector_kerja_jumat,
                axis=1
            )
        if 'collector_kerja_sabtu' not in dt_collector.columns:
            dt_collector['collector_kerja_sabtu'] = dt_collector.apply(
                self._check_collector_kerja_sabtu,
                axis=1
            )
        if 'collector_kerja_minggu' not in dt_collector.columns:
            dt_collector['collector_kerja_minggu'] = dt_collector.apply(
                self._check_collector_kerja_minggu,
                axis=1
            )

        dt_collector.drop(['collector_workday_Sabtu-Minggu','collector_workday_Senin-Jumat',
                           'collector_workday_Senin-Minggu','collector_worktime_Pagi-Siang',
                           'collector_worktime_Pagi-Sore', 'collector_worktime_Siang-Malam',
                           'collector_worktime_Sore-Malam'], axis=1, inplace=True)
        return dt_collector

    def _adjust_education_level(self, dt_collector):
        """
        Adjust the 'dt_collector' DataFrame for education level.

        Parameters:
        - dt_collector (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Modified DataFrame with adjusted education level.
        """
        # Add a new column with the value 0, so that the collector_as_dum has
        # the same dimensions as the debtor_as_dum
        dt_collector['collector_education_level_SD'] = [0] * 1000
        dt_collector['collector_education_level_SMP'] = [0] * 1000

        # Determines the order of the new columns
        new_order = ['collector_age','collector_latitude','collector_longitude',
                     'collector_education_level_D3','collector_education_level_D4',
                    'collector_education_level_S1','collector_education_level_S2',
                    'collector_education_level_S3','collector_education_level_SD',
                    'collector_education_level_SMA','collector_education_level_SMP',
                    'collector_kerja_pagi','collector_kerja_siang','collector_kerja_sore',
                    'collector_kerja_malam','collector_kerja_senin','collector_kerja_selasa',
                    'collector_kerja_rabu','collector_kerja_kamis','collector_kerja_jumat',
                    'collector_kerja_sabtu','collector_kerja_minggu']

        # Use slicing to change the order of columns
        dt_collector = dt_collector[new_order]
        return dt_collector

    def _check_collector_kerja_pagi(self, row):
        """
        Check if the collector has working time in the morning (Pagi).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the morning, 0 otherwise.
        """
        return 1 if (
            (row['collector_worktime_Pagi-Siang'] == 1) or
            (row['collector_worktime_Pagi-Sore'] == 1)
        ) else 0

    def _check_collector_kerja_siang(self, row):
        """
        Check if the collector has working time in the afternoon (Siang).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the afternoon, 0 otherwise.
        """
        return 1 if (
            (row['collector_worktime_Pagi-Siang'] == 1) or
            (row['collector_worktime_Pagi-Sore'] == 1) or
            (row['collector_worktime_Siang-Malam'] == 1)
        ) else 0

    def _check_collector_kerja_sore(self, row):
        """
        Check if the collector has working time in the evening (Sore).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the evening, 0 otherwise.
        """
        return 1 if (
            (row['collector_worktime_Pagi-Sore'] == 1) or
            (row['collector_worktime_Sore-Malam'] == 1)
        )  else 0

    def _check_collector_kerja_malam(self, row):
        """
        Check if the collector has working time in the night (Malam).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the night, 0 otherwise.
        """
        return 1 if (
            (row['collector_worktime_Siang-Malam'] == 1) or
            (row['collector_worktime_Sore-Malam'] == 1)
        ) else 0

    def _check_collector_kerja_senin(self, row):
        """
        Check if the collector has working time in the Monday (Senin).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the Monday, 0 otherwise.
        """
        return 1 if (
            (row['collector_workday_Senin-Jumat'] == 1) or
            (row['collector_workday_Senin-Minggu'] == 1)
        ) else 0

    def _check_collector_kerja_selasa(self, row):
        """
        Check if the collector has working time in the Tuesday (Selasa).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the Tuesday, 0 otherwise.
        """
        return 1 if (
            (row['collector_workday_Senin-Jumat'] == 1) or
            (row['collector_workday_Senin-Minggu'] == 1)
        ) else 0

    def _check_collector_kerja_rabu(self, row):
        """
        Check if the collector has working time in the Wednesday (Rabu).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the Wednesday, 0 otherwise.
        """
        return 1 if (
            (row['collector_workday_Senin-Jumat'] == 1) or
            (row['collector_workday_Senin-Minggu'] == 1)
        ) else 0

    def _check_collector_kerja_kamis(self, row):
        """
        Check if the collector has working time in the Thursday (Kamis).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the Thursday, 0 otherwise.
        """
        return 1 if (
            (row['collector_workday_Senin-Jumat'] == 1) or
            (row['collector_workday_Senin-Minggu'] == 1)
        ) else 0

    def _check_collector_kerja_jumat(self, row):
        """
        Check if the collector has working time in the Friday (Jumat).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the Friday, 0 otherwise.
        """
        return 1 if (
            (row['collector_workday_Senin-Jumat'] == 1) or
            (row['collector_workday_Senin-Minggu'] == 1)
        ) else 0

    def _check_collector_kerja_sabtu(self, row):
        """
        Check if the collector has working time in the Saturday (Sabtu).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the Saturday, 0 otherwise.
        """
        return 1 if (
            (row['collector_workday_Sabtu-Minggu'] == 1) or
            (row['collector_workday_Senin-Minggu'] == 1)
        ) else 0

    def _check_collector_kerja_minggu(self, row):
        """
        Check if the collector has working time in the Sunday (Minggu).

        Parameters:
        - row (pd.Series): A pandas Series representing a row of collector information.

        Returns:
        int: 1 if the collector has working time in the Sunday, 0 otherwise.
        """
        return 1 if (
            (row['collector_workday_Sabtu-Minggu'] == 1) or
            (row['collector_workday_Senin-Minggu'] == 1)
        ) else 0

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's 
    surface using the Haversine formula.

    Parameters:
    - lat1 (float): Latitude of the first point in decimal degrees.
    - lon1 (float): Longitude of the first point in decimal degrees.
    - lat2 (float): Latitude of the second point in decimal degrees.
    - lon2 (float): Longitude of the second point in decimal degrees.

    Returns:
    float: Distance between the two points in kilometers.
    """
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    radius = 6371  # Radius of the Earth in kilometers
    distance = radius * c
    # return distance between the two points in kilometers.
    return distance

def calculate_distance_per_collector(dt_debtor, dt_collector):
    """
    Calculate the distance between a debtor and a list of collectors.

    Parameters:
    - dt_debtor (pd.DataFrame): DataFrame containing debtor information with latitude and longitude.
    - dt_collector (pd.DataFrame): DataFrame containing collector information 
    with latitude and longitude.

    Returns:
    list: List of distances between the debtor and each collector in kilometers.
    """
    dist = []
    for i in range(len(dt_collector)):
        dist.append(calculate_distance(dt_debtor["debtor_latitude"].values[0],
                                    dt_debtor["debtor_longitude"].values[0],
                                    dt_collector["collector_latitude"][i],
                                    dt_collector["collector_longitude"][i]))
    return dist

def scale_for_recsys(dt_debtor, dt_collector):
    """
    Standardize specified features for both debtor and collector DataFrames.

    Parameters:
    - dt_debtor (pd.DataFrame): DataFrame containing debtor information.
    - dt_collector (pd.DataFrame): DataFrame containing collector information.

    Returns:
    tuple: Tuple containing standardized DataFrames for debtor and collector.
    """
    dt_merge = pd.concat([dt_debtor, dt_collector], axis=1)
    features = ['debtor_age', 'debtor_latitude', 'debtor_longitude',
                'collector_age', 'collector_latitude', 'collector_longitude']

    stdscaler = StandardScaler()

    for feature in features:
        dt_merge[feature] = stdscaler.fit_transform(dt_merge[[feature]])

    xdebt = dt_merge.iloc[:,:22]
    xcoll = dt_merge.iloc[:,22:]

    return xdebt, xcoll

class CalculateDistanceFromCoordinates(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate distances between collectors and debtors and 
    add a 'distance' column.

    Attributes:
    - None
    """
    def fit(self, x, y=None):
        """
        Fit method to conform to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.
        - y: Unused parameter.

        Returns:
        self
        """
        return self

    def transform(self, x):
        """
        Transform method to calculate distances between collectors and debtors.

        Parameters:
        - x (pd.DataFrame): Input DataFrame with collector and debtor coordinates.

        Returns:
        pd.DataFrame: DataFrame with added 'distance' column and removed coordinate columns.
        """
        x['distance'] = x.apply(lambda row: calculate_distance(row['collector_latitude'],
                                                               row['collector_longitude'],
                                                               row['debtor_latitude'],
                                                               row['debtor_longitude']),
                                                               axis=1)

        x.drop(['collector_latitude','collector_longitude','debtor_latitude','debtor_longitude'],
               axis=1, inplace=True)
        return x


class TravelingDurationGenerator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to generate traveling duration based on departure and arrival times.

    Attributes:
    - None
    """
    def fit(self, x, y=None):
        """
        Fit method to conform to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.
        - y: Unused parameter.

        Returns:
        self
        """
        return self

    def transform(self, x):
        """
        Transform method to generate traveling duration based on departure and arrival times.

        Parameters:
        - x (pd.DataFrame): Input DataFrame with departure and arrival times.

        Returns:
        pd.DataFrame: DataFrame with added 'traveling_duration' column and 
        removed departure and arrival time columns.
        """
        interactions_df = x[['distance', 'departure_time',
                            'arrival_time', 'transportation_type', 'call_pickup_duration',
                            'door_opening_duration', 'connection_time', 'waiting_response_duration',
                            'idle_duration', 'nonproductive_duration']]
        fmt = '%H:%M'
        interactions_df['departure_time'] = interactions_df['departure_time'].apply(
            lambda x: datetime.strptime(x, fmt)
        )
        interactions_df['arrival_time'] = interactions_df['arrival_time'].apply(
            lambda x: datetime.strptime(x, fmt)
        )
        interactions_df['traveling_duration'] = (
            interactions_df['arrival_time'] - interactions_df['departure_time']
            ).dt.total_seconds().astype(int)

        interactions_df.drop(['departure_time', 'arrival_time'], axis=1, inplace=True)
        return interactions_df


data_preprocessing_best_time = Pipeline([
    ('object_to_categories', ObjectToCategoriesDebtor()),
    ('debt_working_time_transformer', DebtorWorkingTimeTransformer()),
])

data_preprocessing_debtor = Pipeline([
    ('object_to_categories', ObjectToCategoriesDebtor()),
    ('preprocess_for_recsys', PreprocessDebtorForRecSys())
])

data_preprocessing_collector = Pipeline([
    ('object_to_categories', ObjectToCategoriesCollector()),
    ('preprocess_for_recsys', PreprocessCollectorForRecSys())
])

data_preprocessing_input_inter = Pipeline([
    ('calculate_distance_from_coordinates', CalculateDistanceFromCoordinates()),
    ('traveling_duration_generator', TravelingDurationGenerator()),
    ('categorical_encoder', CategoricalEncoder())
])

data_preprocessing_interactions = Pipeline([
    ('traveling_duration_generator', TravelingDurationGenerator()),
    ('categorical_encoder', CategoricalEncoder())
])

scale_and_pca = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('pca', PCA(n_components=3))
])
