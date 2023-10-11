from sklearn.base import BaseEstimator, TransformerMixin
import joblib

class CustomPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        self.model_path = model_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def Pred_cluster_campaign(new_dat, model_path):
            var_vis = 'total_visit'
            var_cal = 'total_calls'
            var_text = 'total_text'
            foreclosure = 'foreclosure'

            # Load the pre-trained model
            loaded_model = joblib.load(model_path)

            visit_campaign_cont = 0
            visit_campaign_beg = 1
            call_campaign = 2

            # Select specific keys from the original dictionary
            expected_features = [var_text, var_cal, var_vis]
            new_data_selected = {key: new_dat[key] for key in expected_features}

            # Predict New data
            new_data_list = [[new_data_selected[feature] for feature in expected_features]]
            predictions = loaded_model.predict(new_data_list)

            predictions = predictions[0]
            campaign_selected = None  # Initialize campaign_selected

            if predictions == visit_campaign_cont:
                if new_data_selected[var_vis] > 4:
                    campaign_selected = 'Lakukan kunjungan Ulang'
                else:
                    if new_dat[foreclosure] == False:
                        campaign_selected = 'Lakukan Penyitaan Barang'
                    elif new_dat[foreclosure] == True:
                        campaign_selected = 'Lakukan Penutupan buku Kreditur'

            elif predictions == visit_campaign_beg:
                campaign_selected = 'Lakukan kunjungan Pertama'

            elif predictions == call_campaign:
                campaign_selected = 'Lakukan Telfon'

            if campaign_selected is None:
                campaign_selected = 'No campaign selected'

            return campaign_selected

        predictions = X.apply(lambda row: Pred_cluster_campaign(row, self.model_path), axis=1)
        result_array = predictions.values
        return result_array

