"""
Module containing APIViews for predicting debtor and collector labels.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from fitur_2.function.feature import predict_debtor_label, predict_collector_label

class DebtorLabel(APIView):
    """
    APIView for predicting debtor labels based on input data.
    """
    def post(self, request):
        """
        Handle POST requests for predicting debtor labels.

        Parameters:
        - request (rest_framework.request.Request): The HTTP request object.

        Returns:
        Response: HTTP response containing the predicted debtor labels.
        """
        try:
            combined_results = predict_debtor_label(request.data)
            return Response(build_result(combined_results.to_dict(orient='records')),
                            status=status.HTTP_200_OK)
        except RuntimeError as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class CollectorLabel(APIView):
    """
    APIView for predicting collector labels based on input data.
    """
    def post(self, request):
        """
        Handle POST requests for predicting collector labels.

        Parameters:
        - request (rest_framework.request.Request): The HTTP request object.

        Returns:
        Response: HTTP response containing the predicted collector labels.
        """
        try:
            combined_results = predict_collector_label(request.data)
            return Response(build_result(combined_results.to_dict(orient='records')),
                            status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

def build_result(combined_result):
    """
    Build a standardized result dictionary.

    Parameters:
    - combined_result (dict): Combined result data to be included in the response.

    Returns:
    dict: Standardized result dictionary.
    """
    message = {
        "status": 200,
        "message": "success",
        "result": combined_result
    }

    return message
