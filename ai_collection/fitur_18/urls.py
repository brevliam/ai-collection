from django.urls import path
from .views import loss_reverse, time_to_collect, total_cost

urlpatterns = [
    path('predict-loss-reverse/', loss_reverse.as_view(), name='Loss_Reverse_Forecast'),
    path('predict-time-to-collect/', time_to_collect.as_view(), name='Loss_Reverse_Forecast'),
    path('predict-total-cost/', total_cost.as_view(), name='Loss_Reverse_Forecast'),

]


