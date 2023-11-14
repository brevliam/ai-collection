"""
Module for Debtor AI Collection Strategy Prediction.

This module provides functionality for predicting the best collection time, method, and collector
in a debtor AI collection strategy. It includes a Django REST Framework APIView class,
`CollectionStrategy`, which handles HTTP requests for predicting the best collection strategy.
Additionally, the module includes a utility function, `build_result`, to construct standardized
result dictionaries for API responses.

Public Objects:
- Class:
    - CollectionStrategy: A Django REST Framework APIView class for predicting the best
                          collection time, method, and collector in a debtor AI collection strategy.

Public Functions:
- build_result: Function to build a standardized result dictionary for API responses.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_3.function.feature import predict_all


class CollectionStrategy(APIView):
    """
    A Django REST Framework APIView class for predicting the best collection time,
    method, and collector in a debtor AI collection strategy.

    Public Methods:
    - post(request): Handles HTTP POST requests for predicting the best collection strategy.
    """
    def post(self, request):
        """
        Handle HTTP POST requests to predict the best collection time, method, and collector
        for a debtor AI collection strategy.

        Parameters:
        - request (Request): The HTTP request object containing the input data.

        Returns:
        - Response: The HTTP response object containing the prediction result.
        """
        try:
            result = predict_all(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message =str(e)
            return Response({"error": error_message}, status=status.HTTP_400_BAD_REQUEST)


def build_result(result):
    """
    Build a standardized result dictionary for API responses.

    Parameters:
    - result (dict): The result dictionary containing the prediction results.

    Returns:
    - dict: A standardized result dictionary with keys:
        - 'status' (int): The HTTP status code (e.g., 200 for success).
        - 'message' (str): A message describing the status or outcome.
        - 'result' (dict): The actual results or predictions.
    """
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result
