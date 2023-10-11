from django.urls import path
from .views import PredictDifficultyScore

urlpatterns = [
    path('predict-difficulty-score/', PredictDifficultyScore.as_view(), name='predict_difficulty_score'),
]