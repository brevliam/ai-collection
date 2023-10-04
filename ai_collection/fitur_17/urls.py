from django.urls import path
from .views import FraudPrediction

urlpatterns = [
    path('predict-fraud/', FraudPrediction.as_view(), name='fraud_prediction'),
]