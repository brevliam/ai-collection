"""
Module for cost-effectiveness prediction views.
"""

from django.shortcuts import render  # pylint: disable=W0611
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_8.function.feature import predict_efficient_human_resources_score

class CostEffectivenessPrediction(APIView):
    """
    API view for cost-effectiveness prediction.
    """
    def post(self, request, format=None):  # pylint: disable=W0622
        """
        Handles POST requests for cost-effectiveness prediction.

        Parameters:
        - request: The HTTP request object.
        - format: The format of the response (default is None).

        Returns:
        - Response: The HTTP response.
        """
        try:
            result = predict_efficient_human_resources_score(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as error_exception:  # pylint: disable=C0103,W0703
            error_message = str(error_exception)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

def build_result(result):
    """
    Builds a standardized result structure.

    Parameters:
    - result: The result data.

    Returns:
    - dict: The standardized result structure.
    """
    result_dict = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result_dict
