from django.urls import path
from .views import DefaultPaymentPrediction

urlpatterns = [
    path('predict-default-payment/', DefaultPaymentPrediction.as_view(), name='test'),
]