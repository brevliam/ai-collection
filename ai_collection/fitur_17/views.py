"""
Module: views.py
Description: This module contains API views for fraud and remedial predictions.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_17.function.feature import predict_fraud_score, predict_remedial_score

def build_result(result):
    """
    Build a standardized result format.

    Args:
        result (dict): The result of the prediction.

    Returns:
        dict: Standardized result format.
    """
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result

class FraudPrediction(APIView):
    """
    API view for predicting fraud scores.
    """
    def post(self, request):
        """
        Handle POST request to predict fraud score.

        Args:
            request (Request): The HTTP request object.

        Returns:
            Response: The HTTP response containing the prediction result.
        """
        try:
            result = predict_fraud_score(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class RemedialPrediction(APIView):
    """
    API view for predicting remedial scores.
    """
    def post(self, request):
        """
        Handle POST request to predict remedial score.

        Args:
            request (Request): The HTTP request object.

        Returns:
            Response: The HTTP response containing the prediction result.
        """
        try:
            result = predict_remedial_score(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

def test():
    """
    Placeholder for testing function.
    """
    return None
