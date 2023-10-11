from django.urls import path
from .views import DebtorLabel, CollectorLabel

urlpatterns = [
    path('debtorlabel/', DebtorLabel.as_view(), name='debtor_label'),
    path('collectorlabel/', CollectorLabel.as_view(), name='collector_label'),
]