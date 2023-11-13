"""
Module containing APIViews for workload prediction, campaign recommendation, 
and field collector recommendation.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from fitur_7.function.feature import predict_workload_score, recommend_campaign,\
    recommend_field_collector

class WorkloadPrediction(APIView):
    """
    APIView for predicting workload scores based on input data.
    """
    def post(self, request):
        """
        Handle POST requests for workload predictions.

        Parameters:
        - request (rest_framework.request.Request): The HTTP request object.

        Returns:
        Response: HTTP response containing the predicted workload scores.
        """
        try:
            result = predict_workload_score(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class CampaignRecommendation(APIView):
    """
    APIView for recommending campaigns based on input data.
    """
    def post(self, request):
        """
        Handle POST requests for campaign recommendations.

        Parameters:
        - request (rest_framework.request.Request): The HTTP request object.

        Returns:
        Response: HTTP response containing the recommended campaigns.
        """
        try:
            result = recommend_campaign(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class FieldCollectorRecommendation(APIView):
    """
    APIView for recommending field collectors based on input data.
    """
    def post(self, request):
        """
        Handle POST requests for field collector recommendations.

        Parameters:
        - request (rest_framework.request.Request): The HTTP request object.

        Returns:
        Response: HTTP response containing the recommended field collectors.
        """
        try:
            result = recommend_field_collector(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

def build_result(result):
    """
    Build a standardized result dictionary.

    Parameters:
    - result (dict): Result data to be included in the response.

    Returns:
    dict: Standardized result dictionary.
    """
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result
