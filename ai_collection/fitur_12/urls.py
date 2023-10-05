from django.urls import path
from .views import recomendation_tenor, request_loan

urlpatterns = [
    path('recommendation-tenor/', recomendation_tenor),
    path('request-loan/', request_loan)
]