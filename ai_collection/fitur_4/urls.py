from django.urls import path
from .views import AssignmentPrediction
from .views import CampaignPrediction


urlpatterns = [
    path('predict-assignment/', AssignmentPrediction.as_view(), name='assignment_prediction'),
    path('predict-campaign/', CampaignPrediction.as_view(), name='campaign_prediction'),
]