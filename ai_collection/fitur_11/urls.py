from django.urls import path
from .views import KreditPinjamanDefaultSolutionPrediction, KreditBendaDefaultSolutionPrediction

urlpatterns = [
    path('predict-default-solution/kredit-pinjaman', KreditPinjamanDefaultSolutionPrediction.as_view(), name='default-kredit-pinjaman'),
    path('predict-default-solution/kredit-benda', KreditBendaDefaultSolutionPrediction.as_view(), name='default-kredit-benda')
    
]