
from django.urls import path
from .views import FraudPrediction, RemedialPrediction

urlpatterns = [
    path('predict-fraud/', FraudPrediction.as_view(), name='fraud_prediction'),
    path('predict-remedial/', RemedialPrediction.as_view(), name='remedial_prediction')
]
