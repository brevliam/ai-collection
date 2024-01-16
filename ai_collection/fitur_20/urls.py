from django.urls import path
from .views import Risk_Status_Prediction

urlpatterns = [
    path('predict-risk_status/', Risk_Status_Prediction.as_view(), name='risk_status_prediction'),
   
]

