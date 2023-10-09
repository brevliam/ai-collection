from django.urls import path
from .views import BestTimetoBillPrediction, RecSysCollectorAssignmentsPrediction, InteractionEfficiencyClusterPrediction
# from . import views

urlpatterns = [
    path('predict-best-time-to-bill/', BestTimetoBillPrediction.as_view(), name='besttimetobill_prediction'),
    path('predict-recommended-collectors-assignments/', RecSysCollectorAssignmentsPrediction.as_view(), name='recsyscollectorassignments_prediction'),
    path('predict-interaction-efficiency-cluster/', InteractionEfficiencyClusterPrediction.as_view(), name='interactioneficiencycluster_prediction')
]