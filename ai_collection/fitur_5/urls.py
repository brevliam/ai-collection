from django.urls import path
from .views import BestTimetoBillPrediction
# from . import views

urlpatterns = [
    path('predict-best-time-to-bill/', BestTimetoBillPrediction.as_view(), name='besttimetobill_prediction')
]