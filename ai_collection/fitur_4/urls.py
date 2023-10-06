from django.urls import path
from .views import AssignmentPrediction
from .views import CampaignPrediciton


urlpatterns = [
    path('predict-assignment/', AssignmentPrediction.as_view(), name='assignment_prediction'),
    path('predict-campaign/', CampaignPrediciton.as_view(), name='campaign_prediction'),
]