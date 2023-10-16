from django.urls import path
from .views import DefaultPaymentPrediction, RecomendSupervision

urlpatterns = [
    path('predict-default-payment/', DefaultPaymentPrediction.as_view(), name='test'),
    path('recommended-supervision/', RecomendSupervision.as_view(), name='recomended')

]