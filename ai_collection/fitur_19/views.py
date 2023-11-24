"""
Module untuk melakukan prediksi terkait fitur-fitur tertentu.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_19.function.feature import (
    predict_default_payment,
    predict_recommended_collectors_supervision
)

def build_result(result):
    """
    Build a standardized result structure.

    Args:
        result (any): The result data to be included in the response.

    Returns:
        dict: A dictionary containing status, message, and result.
    """
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result

class DefaultPaymentPrediction(APIView):
    """
    API view for predicting default payment.

    Methods:
        - post: Handles POST requests for predicting default payment.
    """
    def post(self, request):
        """
        Handle POST requests for predicting default payment.

        Args:
            request (rest_framework.request.Request): The incoming request.

        Returns:
            rest_framework.response.Response: The response containing the prediction result.
        """
        try:
            result = predict_default_payment(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class RecomendSupervision(APIView):
    """
    API view for predicting recommended collectors supervision.

    Methods:
        - post: Handles POST requests for predicting recommended collectors supervision.
    """
    def post(self, request):
        """
        Handle POST requests for predicting recommended collectors supervision.

        Args:
            request (rest_framework.request.Request): The incoming request.

        Returns:
            rest_framework.response.Response: The response containing the prediction result.
        """
        try:
            result = predict_recommended_collectors_supervision(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
