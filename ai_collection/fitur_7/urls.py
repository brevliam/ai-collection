from django.urls import path
from .views import WorkloadPrediction

urlpatterns = [
    path('predict-workload/', WorkloadPrediction.as_view(), name='workload_prediction'),
]