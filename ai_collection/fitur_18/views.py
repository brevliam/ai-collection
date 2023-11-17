"""
Module Docstring:

This module provides Django views for handling loss_reverse, time_to_collect, and total_cost.

Classes:
- `LossReverse`: View class for handling loss_reverse prediction.
- `TimeToCollect`: View class for handling time_to_collect prediction.
- `TotalCost`: View class for handling total_cost prediction.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_18.function.feature import Prediction

def build_result(result):
    """
    Function to build a standardized result dictionary.

    Parameters:
    - result: The result of a prediction.

    Returns:
    A dictionary containing standardized result information.
    """
    result_dict = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result_dict

class LossReverse(APIView):
    """
    View class for handling loss_reverse prediction.
    """

    def post(self, request):
        """
        Handle POST requests for loss_reverse prediction.

        Parameters:
        - request: The HTTP request object.
        - format: The requested response format.

        Returns:
        A Response containing the prediction result.
        """
        try:
            result = Prediction().loss_reverse(request)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class TimeToCollect(APIView):
    """
    View class for handling time_to_collect prediction.
    """

    def post(self, request):
        """
        Handle POST requests for time_to_collect prediction.

        Parameters:
        - request: The HTTP request object.
        - format: The requested response format.

        Returns:
        A Response containing the prediction result.
        """
        try:
            result = Prediction().time_to_collect(request)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class TotalCost(APIView):
    """
    View class for handling total_cost prediction.
    """

    def post(self, request):
        """
        Handle POST requests for total_cost prediction.

        Parameters:
        - request: The HTTP request object.
        - format: The requested response format.

        Returns:
        A Response containing the prediction result.
        """
        try:
            result = Prediction().total_cost(request)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        