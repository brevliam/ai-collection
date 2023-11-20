"""
Module: views
Description: Contains API views for handling feature predictions.
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from fitur_12.function.feature import predict_tenor, predict_loan

class RecommendationTenor(APIView):
    """
    Class: RecommendationTenor
    Description: API endpoint for recommending tenor.

    Methods:
    - post(request): Handles POST requests for tenor recommendation.
    """
    def post(self, request):
        """
        Method: post
        Description: Handles POST requests for tenor recommendation.

        Parameters:
        - request (Request): The incoming HTTP request.

        Returns:
        - Response: JSON response containing the result of tenor prediction.
        """
        try:
            result = predict_tenor(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class RequestLoan(APIView):
    """
    Class: RequestLoan
    Description: Handle POST requests for tenor recommendation.

    Methods:
    - post(request): Handles POST requests for loan prediction.
    """
    def post(self, request):
        """
        Method: post
        Description: Handles POST requests for loan prediction.

        Parameters:
        - request (Request): The incoming HTTP request.

        Returns:
        - Response: JSON response containing the result of loan prediction.
        """
        try:
            result = predict_loan(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

def build_result(result):
    """
    Function: build_result
    Description: Build a standardized result format.

    Parameters:
    - result: The prediction result.

    Returns:
    - dict: A dictionary containing the standardized result format.
    """
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }
    return result
