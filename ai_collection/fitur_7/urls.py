from django.urls import path
from .views import WorkloadPrediction, CampaignRecommendation

urlpatterns = [
    path('predict-workload/', WorkloadPrediction.as_view(), name='workload_prediction'),
    path('recommend-campaign/', CampaignRecommendation.as_view(), name='campaign_recommendation'),
]