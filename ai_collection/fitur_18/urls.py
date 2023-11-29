from django.urls import path
from .views import LossReverse, TimeToCollect, TotalCost

urlpatterns = [
    path('predict-loss-reverse/', LossReverse.as_view(), name='Loss_Reverse_Forecast'),
    path('predict-time-to-collect/', TimeToCollect.as_view(), name='Loss_Reverse_Forecast'),
    path('predict-total-cost/', TotalCost.as_view(), name='Loss_Reverse_Forecast'),

]


