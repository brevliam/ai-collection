"""
    Module containing views for Kredit Pinjaman Default and
    Kredit Benda Default prediction.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_11.function.feature import predict_kredit_pinjaman_default_and_solution, \
                                        predict_kredit_benda_default_and_solution

def build_result(result):
    """
    Build the result dictionary.

    Parameters:
    - result (dict): The result dictionary.

    Returns:
    - dict: The formatted result.
    """
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result

class KreditPinjamanDefaultSolutionPrediction(APIView):
    """
    API view for predicting Kredit Pinjaman Default and Solution.
    """
    def post(self, request):
        """
        Handle POST request for Kredit Pinjaman Default and Solution prediction.

        Parameters:
        - request: The HTTP request.
        - format: The request format (default=None).

        Returns:
        - Response: The API response.
        """
        try:
            result = predict_kredit_pinjaman_default_and_solution(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class KreditBendaDefaultSolutionPrediction(APIView):
    """
    API view for predicting Kredit Benda Default and Solution.
    """
    def post(self, request):
        """
        Handle POST request for Kredit Benda Default and Solution prediction.

        Parameters:
        - request: The HTTP request.
        - format: The request format (default=None).

        Returns:
        - Response: The API response.
        """
        try:
            result = predict_kredit_benda_default_and_solution(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
