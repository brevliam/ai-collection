from django.urls import path
from .views import test, WorkloadPrediction

urlpatterns = [
    path('test/', test, name='test'),
    path('predict-workload/', WorkloadPrediction.as_view(), name='workload_prediction')
]