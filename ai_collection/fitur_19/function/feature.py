"""
Module: AI Collection Rehabilitation

This module provides functions for making predictions related to recommended collectors 
and default payment.

Classes:
    - ObjectToCategoriesDebtor: Custom transformer for converting object columns to categorical 
    columns for debtor data.
    - ObjectToCategoriesCollector: Custom transformer for converting object columns to categorical 
    columns for collector data.
    - PreprocessDebtorForRecSys: Custom transformer for preprocessing debtor data for recommendation 
    system.
    - PreprocessCollectorForRecSys: Custom transformer for preprocessing collector data 
    for recommendation system.

Functions:
    - predict_recommended_collectors_supervision: Predicts recommended collectors for 
    supervision based on input data.
    - predict_default_payment: Predicts default payment based on input data.

Usage:
    Call the provided functions to make predictions related to recommended collectors 
    and default payment.

"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from ..apps import Fitur19Config
from ..libraries import utils

def predict_recommended_collectors_supervision(data):
    """
    Predicts recommended collectors for supervision based on input data.

    Parameters:
    - data (dict): Input data dictionary.

    Returns:
    dict: Result of the prediction.
    """
    model = Fitur19Config.recomended_model
    dataset_file_name_coll = 'collectorr_df.csv'

    input_debtor_df = transform_input(data)
    dataset_coll_path = utils.load_dataset_path(dataset_file_name_coll)
    dt_collector = pd.read_csv(dataset_coll_path)

    preprocessed_debtor = data_preprocessing_debtor.fit_transform(input_debtor_df)
    preprocessed_collector = data_preprocessing_collector.fit_transform(dt_collector)

    output = model.predict([preprocessed_debtor, preprocessed_collector])
    result = transform_recommended_collectors_output(output, dt_collector)

    return result

def predict_default_payment(data):
    """
    Predicts default payment based on input data.

    Parameters:
    - data (dict): Input data dictionary.

    Returns:
    dict: Result of the prediction.
    """
    model = Fitur19Config.default_payment_pred_model
    dataset_file_name = 'debtor_dum.csv'

    input_df = transform_input(data)
    output = model.predict(input_df)
    result = transform_default_payment_pred_output(output)
    # result = output
    utils.append_dataset_with_new_data(dataset_file_name, input_df, result)

    return result

def transform_input(data):
    """
    Transforms input data into a DataFrame.

    Parameters:
    - data (dict): Input data dictionary.

    Returns:
    pd.DataFrame: Transformed DataFrame.
    """
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)

    return df

def transform_default_payment_pred_output(pred):
    """
    Transforms default payment prediction output.

    Parameters:
    - pred (np.ndarray): Model prediction output.

    Returns:
    dict: Transformed result.
    """
    default_payment_next_month = pred[0]

    if default_payment_next_month == 0:
        default_payment_next_month = 'Not Default'
    elif default_payment_next_month == 1:
        default_payment_next_month = 'Default Payment'

    data = {
      'default_payment_next_month': default_payment_next_month,
  }

    return data

class ObjectToCategoriesDebtor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for converting object columns to categorical columns for debtor data.

    Methods:
        - fit(x, y=None): Fit the transformer.
        - transform(x): Transform the input data.

    Attributes:
        None
    """
    def fit(self, x, y=None):
        """
        Fit method conforming to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.
        - y: Unused parameter.

        Returns:
        self
        """
        return self

    def transform(self, x):
        """
        Transform method conforming to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        categorical_columns = ['debtor_education_level', 'action_code']

        x[categorical_columns] = x[categorical_columns].astype('category')

        return x

class ObjectToCategoriesCollector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for converting object columns to categorical columns for collector data.

    Methods:
        - fit(x, y=None): Fit the transformer.
        - transform(x): Transform the input data.

    Attributes:
        None
    """
    def fit(self, x, y=None):
        """
        Fit method conforming to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.
        - y: Unused parameter.

        Returns:
        self
        """
        return self

    def transform(self, x):
        """
        Transform method conforming to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        categorical_columns = ['collector_education_level', 'action_code_history']

        x[categorical_columns] = x[categorical_columns].astype('category')

        return x


class PreprocessDebtorForRecSys(BaseEstimator, TransformerMixin):
    """
    Custom transformer for preprocessing debtor data for recommendation system.

    Methods:
        - fit(x, y=None): Fit the transformer.
        - transform(x): Transform the input data.

    Attributes:
        None
    """
    def fit(self, x, y=None):
        """
        Fit method conforming to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.
        - y: Unused parameter.

        Returns:
        self
        """
        return self

    def transform(self, x):
        """
        Transform method conforming to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        debt_edulvl = self._transform_debt_edu_level(x['debtor_education_level'])
        debt_personality = self._transform_debt_personality(x['debtor_personality'])
        action_code = self._transform_debt_action_code(x['action_code'])
        dt_debtor = pd.concat([debt_edulvl, debt_personality, action_code], axis=1)
        dt_debtor = pd.concat([dt_debtor] * 1000)

        return dt_debtor.reset_index(drop=True)

    def _transform_debt_edu_level(self, debt_edu_level):
        """
        Transform debtor education level into one-hot encoded DataFrame.

        Parameters:
        - debt_edu_level (pd.Series): Categorical values of debtor education level.

        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        debt_edu_level_dum = []
        if debt_edu_level.values[0] == 'D3':
            debt_edu_level_dum.append(1)
        else:
            debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'S1':
            debt_edu_level_dum.append(1)
        else:
            debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'S2':
            debt_edu_level_dum.append(1)
        else:
            debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'SD':
            debt_edu_level_dum.append(1)
        else:
            debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'SMA':
            debt_edu_level_dum.append(1)
        else:
            debt_edu_level_dum.append(0)
        if debt_edu_level.values[0] == 'SMP':
            debt_edu_level_dum.append(1)
        else:
            debt_edu_level_dum.append(0)


        debt_edu_dum = np.array(debt_edu_level_dum).reshape((1, 6))

        debt_edu_df = pd.DataFrame(debt_edu_dum,
                                   columns=['debtor_education_level_D3','debtor_education_level_S1',
                                            'debtor_education_level_S2','debtor_education_level_SD',
                                            'debtor_education_level_SMA',
                                            'debtor_education_level_SMP'])

        return debt_edu_df

    def _transform_debt_personality(self, debt_personality):
        """
        Transform debtor personality into one-hot encoded DataFrame.

        Parameters:
        - debt_personality (pd.Series): Categorical values of debtor personality.

        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        debt_personality_dum = []
        if debt_personality.values[0] == 'kalem':
            debt_personality_dum.append(1)
        else:
            debt_personality_dum.append(0)
        if debt_personality.values[0] == 'sedang':
            debt_personality_dum.append(1)
        else:
            debt_personality_dum.append(0)


        debt_pers_dum = np.array(debt_personality_dum).reshape((1, 2))

        debt_pers_df = pd.DataFrame(debt_pers_dum, columns=['debtor_personality_kalem',
                                                            'debtor_personality_sedang'])

        return debt_pers_df

    def _transform_debt_action_code(self, debt_action_code):
        """
        Transform debtor action code into one-hot encoded DataFrame.

        Parameters:
        - debt_action_code (pd.Series): Categorical values of debtor action code.

        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        debt_action_code_dum = []
        if debt_action_code.values[0] == 'ATPU':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'BUSY':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'CBLT':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'DISC':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'DISP':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'FDBK':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'HUPD':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'INCO':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'LMSG':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'MUPD':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'NOAN':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'NOCO':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'OTHR':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'PTPR':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'PTPY':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'WRPH':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)
        if debt_action_code.values[0] == 'WUPD':
            debt_action_code_dum.append(1)
        else:
            debt_action_code_dum.append(0)


        debt_act_code_dum = np.array(debt_action_code_dum).reshape((1, 17))

        debt_act_code_df  = pd.DataFrame(debt_act_code_dum, columns=['action_code_ATPU',
                              'action_code_BUSY',
                              'action_code_CBLT',
                              'action_code_DISC',
                              'action_code_DISP',
                              'action_code_FDBK',
                              'action_code_HUPD',
                              'action_code_INCO',
                              'action_code_LMSG',
                              'action_code_MUPD',
                              'action_code_NOAN',
                              'action_code_NOCO',
                              'action_code_OTHR',
                              'action_code_PTPR',
                              'action_code_PTPY',
                              'action_code_WRPH',
                              'action_code_WUPD'])

        return debt_act_code_df



class PreprocessCollectorForRecSys(BaseEstimator, TransformerMixin):
    """
    Custom transformer for preprocessing collector data for the recommendation system.

    Methods:
        - fit(x, y=None): Fit the transformer.
        - transform(x): Transform the input data.

    Attributes:
        None
    """
    def fit(self, x, y=None):
        """
        Fit method conforming to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.
        - y: Unused parameter.

        Returns:
        self
        """
        return self

    def transform(self, x):
        """
        Transform method conforming to the sklearn transformer interface.

        Parameters:
        - x (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        dt_collector = x[['collector_education_level','collector_personality',
                          'action_code_history']]

        dt_collector = pd.get_dummies(dt_collector, columns=['collector_education_level',
                                                             'collector_personality',
                                                             'action_code_history'])

        return dt_collector.reset_index(drop=True)

data_preprocessing_debtor = Pipeline([
    ('object_to_categories', ObjectToCategoriesDebtor()),
    ('preprocess_for_recsys', PreprocessDebtorForRecSys())
])

data_preprocessing_collector = Pipeline([
    ('object_to_categories', ObjectToCategoriesCollector()),
    ('preprocess_for_recsys', PreprocessCollectorForRecSys())
])

def transform_recommended_collectors_output(output, dt_collector):
    output = pd.DataFrame(output)
    output.sort_values(by=[0], ascending=[False], inplace=True)
    recommended_collectors_index = output[:5].index
    recommended_collectors = (
        dt_collector
        .loc[recommended_collectors_index, 'collector_name']
        .tolist()
    )

    result = {
        'recommended_collectors_to_monitor': recommended_collectors
    }

    return result
