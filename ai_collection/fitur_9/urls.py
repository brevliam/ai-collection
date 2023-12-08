from django.urls import path
from .views import KreditRecommendedPrediction 
urlpatterns = [
    path('predict-recomended-solution/kredit-pinjaman', KreditRecommendedPrediction.as_view(), name='default-kredit-pinjaman')  
]