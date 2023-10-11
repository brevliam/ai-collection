from django.urls import path
from .views import KreditRecomendedPrediction #, KreditBendaDefaultSolutionPrediction

urlpatterns = [
    path('predict-recomended-solution/kredit-pinjaman', KreditRecomendedPrediction.as_view(), name='default-kredit-pinjaman')  
]