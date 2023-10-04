from django.urls import path
from .views import predict_difficulty_score

urlpatterns = [
    path('predict-difficulty-score/', predict_difficulty_score, name='predict_difficulty_score'),
]