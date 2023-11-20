"""
    This module contains every method for inference,
    as well as data transformation and
    preprocessing.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from ..apps import Fitur11Config
from ..libraries import utils



class LoanIncomeExpensesRatioCalculator(BaseEstimator, TransformerMixin):
    """
    Transformer to calculate the loan income expenses ratio.

    Methods:
    - fit(self, data): Fit the transformer.
    - transform(self, data): Transform the input data.

    Attributes:
    - None
    """
    def fit(self, data):
        """
            Fit the transformer.

            Parameters:
            - data: Input data.
            - y: Target data (default=None).

    Returns:
    - self
        """
        return self

    def transform(self, data):
        """
            Transform the input data by calculating
            the loan-to-income-and-expenses ratio.

            Parameters:
            - data: Input data containing 'monthly_payment', 'monthly_income',
            and 'monthly_expenses' columns.

            Returns:
            - Transformed data with an added 'loan_income_expenses_ratio' column.
        """
        data['loan_income_expenses_ratio'] = round(
            data['monthly_payment'] / (
                data['monthly_income'] - data['monthly_expenses']
                ) * 100, 2
        )
        return data


class DefaultRiskCalculator(BaseEstimator, TransformerMixin):
    """
        Generate Default Risk
    """
    def fit(self, data):
        """
        Fit the transformer.

        Parameters:
        - data: Input data.
        - y: Target data (default=None).

        Returns:
        - self
        """
        return self

    def transform(self, data):
        """
        Transform the input data by calculating the default risk.

        Parameters:
        - data: Input data.

        Returns:
        - Transformed data with an added 'default_risk' column.
        """
        data['default_risk'] = data.apply(self.calculate_default_risk, axis=1)
        return data

    def calculate_default_risk(self, row):
        """
        Calculate the default risk based on
        loan-to-income-and-expenses ratio and other factors.

        Parameters:
        - row: A row of the input data.

        Returns:
        - Default risk category.
        """

        risk_category = None

        if row['loan_income_expenses_ratio'] < 20:
            risk_category = "Sangat Baik"

        elif 20 <= row['loan_income_expenses_ratio'] < 40:
            if row['asset_value'] * 0.2 > row['loan_amount']:
                risk_category = "Sangat Baik"
            elif row['asset_value'] * 0.5 > row['loan_amount']:
                risk_category = "Baik"
            else:
                risk_category = "Netral"

        elif 40 <= row['loan_income_expenses_ratio'] < 60:
            if row['asset_value'] * 0.2 >= row['loan_amount']:
                risk_category = "Baik"
            elif row['asset_value'] * 0.5 >= row['loan_amount']:
                risk_category = "Beresiko"
            else:
                risk_category = "Beresiko"

        elif 60 <= row['loan_income_expenses_ratio'] <= 80:
            if row['asset_value'] * 0.2 >= row['loan_amount']:
                risk_category = "Beresiko"
            elif row['asset_value'] * 0.5 <= row['loan_amount']:
                risk_category = "Sangat Beresiko"

        elif row['loan_income_expenses_ratio'] >= 80:
            risk_category = "Sangat Beresiko"

        elif row['loan_income_expenses_ratio'] >= 60:
            risk_category = "Beresiko"

        return risk_category

class SESCalculator(BaseEstimator, TransformerMixin):
    """ Calculate SES """
    def fit(self, data):
        """
        Fit the transformer.

        Parameters:
        - data: Input data.
        Returns:
        - self
        """
        return self

    def transform(self, data):
        """
        Transform the input data by calculating the SES (Socioeconomic Status).

        Parameters:
        - data: Input data.

        Returns:
        - Transformed data with an added 'ses' column.
        """
        data['ses'] = data.apply(self.calculate_ses, axis=1)
        return data

    def calculate_ses(self, row):
        """
        Calculate the SES based on debtor's education level, monthly income,
        asset value, and monthly expenses.

        Parameters:
        - row: A row of the input data.

        Returns:
        - SES category.
        """
        result = "Tidak Diketahui"  # Default value for unknown education level

        if row['debtor_education_level'] == "SMA":
            if (
                row['monthly_income'] < 3000000 or
                row['asset_value'] < 50000000 or
                row['monthly_expenses'] > 0.5 * row['monthly_income']
            ):
                result = "Sangat Rendah"
            elif (
                row['monthly_income'] < 8000000 and
                row['asset_value'] < 400000000 and
                row['monthly_expenses'] <= 0.6 * row['monthly_income']
            ):
                result = "Rendah"
            elif (
                row['monthly_income'] < 15000000 and
                row['asset_value'] < 800000000 and
                row['monthly_expenses'] <= 0.7 * row['monthly_income']
            ):
                result = "Menengah"
            elif (
                row['monthly_income'] < 30000000 and
                row['asset_value'] < 1000000000 and
                row['monthly_expenses'] <= 0.8 * row['monthly_income']
            ):
                result = "Tinggi"
            else:
                result = "Sangat Tinggi"
        elif row['debtor_education_level'] == "D3" or row['debtor_education_level'] == "D4":
            if (
                row['monthly_income'] < 3500000 or
                row['asset_value'] < 50000000 or
                row['monthly_expenses'] > 0.5 * row['monthly_income']
            ):
                result = "Sangat Rendah"
            elif (
                row['monthly_income'] < 10000000 and
                row['asset_value'] < 300000000 and
                row['monthly_expenses'] <= 0.6 * row['monthly_income']
            ):
                result = "Rendah"
            elif (
                row['monthly_income'] < 20000000 and
                row['asset_value'] < 600000000 and
                row['monthly_expenses'] <= 0.7 * row['monthly_income']
            ):
                result = "Menengah"
            elif (
                row['monthly_income'] < 40000000 and
                row['asset_value'] < 800000000 and
                row['monthly_expenses'] <= 0.8 * row['monthly_income']
            ):
                result = "Tinggi"
            else:
                result = "Sangat Tinggi"
        elif row['debtor_education_level'] == "S1":
            if (
                row['monthly_income'] < 5000000 or
                row['asset_value'] < 50000000 or
                row['monthly_expenses'] > 0.5 * row['monthly_income']
            ):
                result = "Sangat Rendah"
            elif (
                row['monthly_income'] < 13000000 and
                row['asset_value'] < 300000000 and
                row['monthly_expenses'] <= 0.6 * row['monthly_income']
            ):
                result = 'Rendah'
            elif (
                row['monthly_income'] < 26000000 and
                row['asset_value'] < 600000000 and
                row['monthly_expenses'] <= 0.7 * row['monthly_income']
            ):
                result = "Menengah"
            elif (
                row['monthly_income'] < 52000000 and
                row['asset_value'] < 1000000000 and
                row['monthly_expenses'] <= 0.8 * row['monthly_income']
            ):
                result = "Tinggi"
            else:
                result = "Sangat Tinggi"

        return result



class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
        Custom transformer for encoding categorical features into
        numerical valuesbased on predefined mappings.

        Parameters:
        None

        Attributes:
        - mapping (dict): A dictionary containing mappings for each categorical feature.

        Methods:
        - fit(X): Fit the transformer. Since the transformer
        does not learn from the data, this method returns self.

        - transform(X): Transform the input DataFrame X by applying
        the predefined mappings to the categorical columns.

        Examples:
        ```
        # Create an instance of CategoricalEncoder
        encoder = CategoricalEncoder()

        # Fit and transform a DataFrame
        X_encoded = encoder.fit_transform(X_categorical)
        ```
    """
    def __init__(self):
        """
            Initialize the CategoricalEncoder with predefined mappings
            for various categorical features.
        """
        self.mapping = {
            "debtor_education_level": {
                "SD": 0, "SMP": 1, "SMA": 2, "D3": 3, "D4": 4, "S1": 5, "S2": 6, "S3": 7
            },
            "debtor_marital_status": {
                "Belum menikah": 0, "Sudah menikah": 1, "Cerai Hidup": 2, "Cerai Mati": 3
            },
            "ses": {
                "Sangat Rendah": 0, "Rendah": 1, "Menengah": 2, "Tinggi": 3, "Sangat Tinggi": 4
            },
            "clasify_total_capital": {
                "Sangat Kuat": 1, "Cukup": 2, "Lemah": 3, "Kuat": 4, "Sangat Lemah": 0
            },
            "default_risk": {
                "Baik": 1, "Sangat Baik": 0, "Netral": 2, "Beresiko": 3, "Sangat Beresiko": 4
            },
            "loan_purpose": {
                "kredit Modal": 1, "Kebutuhan darurat": 2, "kredit pribadi": 3,
                "pernikahan": 4, "lainnya": 0, "kredit properti": 1,
                "kendaraan bermotor": 0
            },
            "default_potential" : {"Sangat Baik" : 0,  "Baik" : 1, "Netral" : 2,
                                   "Buruk" : 3, "Sangat Buruk" : 4,
                                   "Suspicious" : 5}
        }

    def fit(self, data):
        """
        Fit the transformer.

        Parameters:
        - data (DataFrame): Input data.
        - y (array-like, optional): Target values (ignored).

        Returns:
        - self: Returns an instance of the transformer.
        """
        return self

    def transform(self, data):
        """
        Transform the input DataFrame data by applying the predefined
        mappings to the categorical columns.

        Parameters:
        - X (DataFrame): Input data.

        Returns:
        - X (DataFrame): Transformed data with categorical columns
        replaced by numerical values.
        """
        with pd.option_context('mode.chained_assignment', None):
            for column, mapping in self.mapping.items():
                if column in data.columns:
                    data[column] = data[column].map(mapping)
        return data


class PinjamanPotentialDefaultCalculator(BaseEstimator, TransformerMixin):
    """
    Custom transformer for calculating potential default categories
    based on a default score.

    Parameters:
    None

    Methods:
    - fit(X): Fit the transformer. Since the transformer does
    not learn from the data, this method returns self.

    - transform(X): Transform the input DataFrame X by calculating and
    adding a 'default_potential' column.

    - calculate_default_potential(row): Calculate the default potential category
    based on the 'default_score' in the input row.

    Examples:
    ```
    # Create an instance of PinjamanPotentialDefaultCalculator
    calculator = PinjamanPotentialDefaultCalculator()

    # Fit and transform a DataFrame
    X_transformed = calculator.fit_transform(X)
    ```

    Attributes:
    None
    """
    def fit(self, data):
        """
        Fit the transformer.

        Parameters:
        - data (DataFrame): Input data.
        - y (array-like, optional): Target values (ignored).

        Returns:
        - self: Returns an instance of the transformer.
        """
        return self

    def transform(self, data):
        """
        Transform the input DataFrame data by calculating and
        adding a 'default_potential' column.

        Parameters:
        - data (DataFrame): Input data.

        Returns:
        - data (DataFrame): Transformed data with an added
        'default_potential' column.
        """
        data['default_potential'] = data.apply(self.calculate_default_potential, axis=1)
        return data

    def calculate_default_potential(self, row):
        """
        Calculate the default potential category based on
        the 'default_score' in the input row.

        Parameters:
        - row (Series): A row of the input data.

        Returns:
        - str: Default potential category.
        """
        if row['default_score'] <= 100:
            return 'Sangat Baik'
        if row['default_score'] <= 250:
            return 'Baik'
        if row['default_score'] <= 500:
            return 'Netral'
        if row['default_score'] <= 700:
            return 'Buruk'
        if row['default_score'] <= 850:
            return 'Sangat Buruk'
        return 'Suspicious'
class BendaPotentialDefaultCalculator(BaseEstimator, TransformerMixin):
    """
    Transformer for calculating potential default for the
    'Benda' class based on the default score.

    Attributes:
    - None

    Methods:
    - fit(self): Fit the transformer.
    - transform(self, data): Transform the input data.
    - calculate_default_potential(self, row): Calculate the default potential
    based on the default score.
    """
    def fit(self, data):
        """
        Fit the transformer.

        Parameters:
        - data: Input data.
        - y: Target data (default=None).

        Returns:
        - self
        """
        return self

    def transform(self, data):
        """
        Transform the input data.

        Parameters:
        - data: Input data.

        Returns:
        - Transformed data.
        """
        data['default_potential'] = data.apply(self.calculate_default_potential, axis=1)
        return data

    def calculate_default_potential(self, row):
        """
        Calculate the default potential based on the default score.

        Parameters:
        - row: A row of the input data.

        Returns:
        - Default potential category.
        """
        if row['default_score'] <= 100:
            return 'Sangat Baik'
        if row['default_score'] <= 250:
            return 'Baik'
        if row['default_score'] <= 550:
            return 'Netral'
        if row['default_score'] <= 750:
            return 'Buruk'
        if row['default_score'] <= 900:
            return 'Sangat Buruk'
        return 'Suspicious'


data_transformation_pipeline = Pipeline([
    ('loan_income_expenses_ratio', LoanIncomeExpensesRatioCalculator()),
    ('default_risk_calculator', DefaultRiskCalculator()),
    ('ses_calculator', SESCalculator()),
])


categorical_preprocessing = Pipeline([
    ('categorical_encoder', CategoricalEncoder()),
])



def predict_kredit_pinjaman_default_and_solution(data):
    """
    Predicts Kredit Pinjaman Default and Solution based on the provided input data.

    Parameters:
    - data (dict): Input data for prediction.

    Returns:
    - dict: Dictionary containing the predicted default score, default potential, and solution.
    """
    default_kredit_model = Fitur11Config.default_kredit_model
    default_kredit_scaler = Fitur11Config.default_kredit_scaler
    default_solution_model = Fitur11Config.default_solution_model
    default_solution_scaler = Fitur11Config.default_solution_scaler
    default_kredit_dataset = "kredit_pinjaman_dataset.csv"
    data_df = transform_input(data)
    data_transformation_pipeline.fit_transform(data_df)
    new_column_order = ['employment_year', 'monthly_income', 'monthly_expenses', 'asset_value',
                            'loan_amount', 'interest_rate', 'tenor', 'monthly_payment',
                            'loan_income_expenses_ratio', 'default_risk', 'loan_purpose']
    kredit_df = data_df[new_column_order]
    categorical_preprocessing.fit_transform(kredit_df)
    kredit_df = default_kredit_scaler.transform(kredit_df)
    default_score = default_kredit_model.predict(kredit_df)
    data_df['default_score'] = default_score
    default_potential_cal = PinjamanPotentialDefaultCalculator()
    default_potential_cal.fit_transform(data_df)
    solution_df = data_df[['debtor_age', 'asset_value', 'loan_amount', 'loan_purpose',
                      'ses','default_risk', 'default_score', 'default_potential']]
    categorical_preprocessing.fit_transform(solution_df)
    solution_df = default_solution_scaler.transform(solution_df)
    data_df['solution'] = default_solution_model.predict(solution_df)
    utils.append_new_row(default_kredit_dataset, data_df)
    data = {
        "default_score": round(data_df['default_score'].values[0]),
        "default_potential": data_df['default_potential'].values[0],
        "solution": data_df['solution'].values[0],
    }
    return data
def predict_kredit_benda_default_and_solution(data):
    """
    Predicts Kredit Benda Default and Solution based on the provided input data.

    Parameters:
    - data (dict): Input data for prediction.

    Returns:
    - dict: Dictionary containing the predicted default score, default potential, and solution.
    """
    default_kredit_model = Fitur11Config.default_kredit_benda_model
    default_kredit_scaler = Fitur11Config.default_kredit_benda_scaler
    default_solution_model = Fitur11Config.default_solution_benda_model
    default_solution_scaler = Fitur11Config.default_solution_benda_scaler
    default_kredit_dataset = "kredit_benda_dataset.csv"
    data_df = transform_input(data)
    data_transformation_pipeline.fit_transform(data_df)
    new_column_order = ['employment_year', 'monthly_income', 'monthly_expenses', 'asset_value',
                            'loan_amount', 'interest_rate', 'tenor', 'monthly_payment',
                            'loan_income_expenses_ratio', 'default_risk', 'ses']


    kredit_df = data_df[new_column_order]
    categorical_preprocessing.fit_transform(kredit_df)
    kredit_df = default_kredit_scaler.transform(kredit_df)
    default_score = default_kredit_model.predict(kredit_df)
    data_df['default_score'] = default_score
    default_potential_cal = BendaPotentialDefaultCalculator()
    default_potential_cal.fit_transform(data_df)
    solution_df  = data_df[['debtor_age', 'asset_value', 'loan_amount', 'loan_purpose',
                       'ses','default_risk', 'default_score', 'default_potential']]
    categorical_preprocessing.fit_transform(solution_df)
    solution_df = default_solution_scaler.transform(solution_df)
    data_df['solution'] = default_solution_model.predict(solution_df)
    utils.append_new_row(default_kredit_dataset, data_df)
    data = {
        "default_score": round(data_df['default_score'].values[0]),
        "default_potential": data_df['default_potential'].values[0],
        "solution": data_df['solution'].values[0],
    }
    return data
def transform_input(data):
    """
    Transforms the input data into a pandas DataFrame.

    Parameters:
    - data (dict): Input data to be transformed.

    Returns:
    - pd.DataFrame: Transformed DataFrame.
    """
    data = {key: [value] for key, value in data.items()}
    data_df = pd.DataFrame(data)
    return data_df
