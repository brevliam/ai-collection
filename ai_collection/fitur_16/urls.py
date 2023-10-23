from django.urls import path
from .views import AuctionfraudPrediction, CollateralAppraisal


urlpatterns = [
    path('predict-fraud/', AuctionfraudPrediction.as_view(), name='fraud_prediction'),
    path('predict-appraisal/', CollateralAppraisal.as_view(), name='appraisal_prediction'),
    
]

