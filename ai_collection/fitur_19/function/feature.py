from ..apps import Fitur19Config
from ..libraries import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

import pandas as pd

def predict_recommended_collectors_supervision(data):
    model = Fitur19Config.recomended_model
    DATASET_FILE_NAME_COLL = 'collectorr_df.csv'

    input_debtor_df = transform_input(data)
    dataset_coll_path = utils.load_dataset_path(DATASET_FILE_NAME_COLL)
    dt_collector = pd.read_csv(dataset_coll_path)

    preprocessed_debtor = data_preprocessing_debtor.fit_transform(input_debtor_df)
    preprocessed_collector = data_preprocessing_collector.fit_transform(dt_collector)

    output = model.predict([preprocessed_debtor, preprocessed_collector])
    result = transform_recommended_collectors_output(output, input_debtor_df, dt_collector)
    
    return result

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

class ObjectToCategoriesDebtor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categorical_columns = ['debtor_education_level', 'action_code']

        X[categorical_columns] = X[categorical_columns].astype('category')

        return X

class ObjectToCategoriesCollector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categorical_columns = ['collector_education_level', 'action_code_history']

        X[categorical_columns] = X[categorical_columns].astype('category')

        return X


class PreprocessDebtorForRecSys(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        debt_edulvl = self._transform_debt_edu_level(X['debtor_education_level'])
        debt_personality = self._transform_debt_personality(X['debtor_personality'])
        action_code = self._transform_debt_action_code(X['action_code'])
        dt_debtor = pd.concat([debt_edulvl, debt_personality, action_code], axis=1)
        dt_debtor = pd.concat([dt_debtor] * 1000)

        return dt_debtor.reset_index(drop=True)

    def _transform_debt_edu_level(self, debt_edu_level):
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

        debt_edu_df = pd.DataFrame(debt_edu_dum, columns=['debtor_education_level_D3','debtor_education_level_S1',
                                                          'debtor_education_level_S2','debtor_education_level_SD',
                                                          'debtor_education_level_SMA','debtor_education_level_SMP'])

        return debt_edu_df

    def _transform_debt_personality(self, debt_personality):
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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt_collector = X[['collector_education_level','collector_personality','action_code_history']]

        dt_collector = pd.get_dummies(dt_collector, columns=['collector_education_level',
                                                      'collector_personality','action_code_history'])

        return dt_collector.reset_index(drop=True)

   

data_preprocessing_debtor = Pipeline([
    ('object_to_categories', ObjectToCategoriesDebtor()),
    ('preprocess_for_recsys', PreprocessDebtorForRecSys())
])

data_preprocessing_collector = Pipeline([
    ('object_to_categories', ObjectToCategoriesCollector()),
    ('preprocess_for_recsys', PreprocessCollectorForRecSys())
])

def transform_recommended_collectors_output(output, input_debtor_df, dt_collector):
    output = pd.DataFrame(output)
    output.sort_values(by=[0], ascending=[False], inplace=True)
    recommended_collectors_index = output[:5].index
    recommended_collectors = dt_collector.loc[recommended_collectors_index, 'collector_name'].tolist()
    result = {
        'recommended_collectors_to_monitor': recommended_collectors
    }

    return result