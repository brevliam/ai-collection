from django.urls import path
from .views import CostEffectivenessPrediction
urlpatterns = [
    path('predict-CostEffectivenessPrediction/', CostEffectivenessPrediction.as_view(), name='CostEffectiveness_Prediction'),
]

