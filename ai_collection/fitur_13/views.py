from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from fitur_13.function.feature import predict_best_time_to_remind, predict_best_time_to_follow_up, predict_reschedule


class BestTimetoRemindPrediction(APIView):
    def post(self, request, format=None):
        try:
            result = predict_best_time_to_remind(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = {
            "error" : str(e),
            "status": status.HTTP_400_BAD_REQUEST
            }
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)
        
class BestTimetoFollowUpPrediction(APIView):
    def post(self, request, format=None):
        try:
            result = predict_best_time_to_follow_up(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = {
            "error" : str(e),
            "status": status.HTTP_400_BAD_REQUEST
            }
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)

class ReschedulePrediction(APIView):
    def post(self, request, format=None):
        try:
            result = predict_reschedule(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = {
            "error" : str(e),
            "status": status.HTTP_400_BAD_REQUEST
            }
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)
    
def build_result(result):
    message = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return message