from django.urls import path
from .views import BestTimetoRemindPrediction, BestTimetoFollowUpPrediction, ReschedulePrediction
# from . import views

urlpatterns = [
    path('predict-best-time-to-remind/', BestTimetoRemindPrediction.as_view(), name='best_time_to_remind_prediction'),
    path('predict-best-time-to-follow-up/', BestTimetoFollowUpPrediction.as_view(), name='best_time_to_follow_up_prediction'),
    path('predict-reschedule/', ReschedulePrediction.as_view(), name='best_time_to_follow_up_prediction')


]