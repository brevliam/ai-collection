from django.urls import path
from .views import CollectionStrategy

urlpatterns = [
    path('predict-collection-strategy/', CollectionStrategy.as_view(), name='collection_strategy_prediction'),
]