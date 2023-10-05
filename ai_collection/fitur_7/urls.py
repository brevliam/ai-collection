from django.urls import path
from .views import WorkloadPrediction, CampaignRecommendation, FieldCollectorRecommendation

urlpatterns = [
    path('predict-workload/', WorkloadPrediction.as_view(), name='workload_prediction'),
    path('recommend-campaign/', CampaignRecommendation.as_view(), name='campaign_recommendation'),
    path('recommend-field-collector/', FieldCollectorRecommendation.as_view(), name='field_collector_recommendation')
]