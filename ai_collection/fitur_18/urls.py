from django.urls import path
from .views import loss_reverse#, credit_risk , time_to_collect, total_cost

urlpatterns = [
    path('predict-loss_reverse/', loss_reverse.as_view(), name='Loss_Reverse_Forecast'),
    # path('predict-credit_risk/', credit_risk.as_view(), name='Loss_Reverse_Forecast'),
    # path('predict-time_to_collect/', time_to_collect.as_view(), name='Loss_Reverse_Forecast'),
    # path('predict-total_cost/', total_cost.as_view(), name='Loss_Reverse_Forecast'),

]


