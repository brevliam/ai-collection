"""
Module-level of views.py.
"""
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from fitur_13.function.feature import (
    predict_best_time_to_remind,
    predict_best_time_to_follow_up,
    predict_reschedule,
)


class BestTimetoRemindPrediction(APIView):
    """
    API endpoint for predicting the best time to remind.

    Accepts POST requests with data to make predictions.
    Returns the prediction result in the response.
    """

    def post(self, request):
        """
        Handle POST requests for predicting the best time to remind.

        Parameters:
        - request: The incoming HTTP request.

        Returns:
        - Response: HTTP response containing the prediction result.
        """
        try:
            result = predict_best_time_to_remind(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = {
                "error": str(e),
                "status": status.HTTP_400_BAD_REQUEST,
            }
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)


class BestTimetoFollowUpPrediction(APIView):
    """
    API endpoint for predicting the best time to follow up.

    Accepts POST requests with data to make predictions.
    Returns the prediction result in the response.
    """

    def post(self, request):
        """
        Handle POST requests for predicting the best time to follow up.

        Parameters:
        - request: The incoming HTTP request.

        Returns:
        - Response: HTTP response containing the prediction result.
        """
        try:
            result = predict_best_time_to_follow_up(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = {
                "error": str(e),
                "status": status.HTTP_400_BAD_REQUEST,
            }
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)


class ReschedulePrediction(APIView):
    """
    API endpoint for predicting reschedule eligibility.

    Accepts POST requests with data to make predictions.
    Returns the prediction result in the response.
    """

    def post(self, request):
        """
        Handle POST requests for predicting reschedule eligibility.

        Parameters:
        - request: The incoming HTTP request.

        Returns:
        - Response: HTTP response containing the prediction result.
        """
        try:
            result = predict_reschedule(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = {
                "error": str(e),
                "status": status.HTTP_400_BAD_REQUEST,
            }
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)


def build_result(result):
    """
    Build a standardized result message.

    Parameters:
    - result: The result to be included in the message.

    Returns:
    - dict: A dictionary containing a standardized result message.
    """
    message = {
        "status": 200,
        "message": "success",
        "result": result,
    }
    return message