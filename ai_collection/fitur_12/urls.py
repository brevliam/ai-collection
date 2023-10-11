from django.urls import path
from .views import RecommendationTenor, RequestLoan

urlpatterns = [
    path('recommendation-tenor/', RecommendationTenor.as_view(), 
         name='recommendation_tenor'),
    path('request-loan/', RequestLoan.as_view(), 
         name='request_loan'),
]