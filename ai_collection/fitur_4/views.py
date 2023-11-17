"""
Module for Django views using the REST framework.

This module includes views for rendering 
Django templates and handling API requests using the Django REST framework.

Imports:
    - `render` from `django.shortcuts`: A function for rendering Django templates.
    - `APIView` from `rest_framework.views`: A class for creating class-based views.
    - `Response` from `rest_framework.response`: A class for handling API responses.
    - `status` from `rest_framework`: Constants representing HTTP status codes.

Example:
    >>> from django.shortcuts import render
    >>> from rest_framework.views import APIView
    >>> from rest_framework.response import Response
    >>> from rest_framework import status

See Also:
    - [Django documentation]
    (https://docs.djangoproject.com/) for Django usage.
    - [Django REST framework documentation]
    (https://www.django-rest-framework.org/) for REST framework usage.
"""
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


from fitur_4.function.feature import predict_assignment
from fitur_4.function.feature import predict_campaign


class AssignmentPrediction(APIView):
    """
    API endpoint for predicting assignments.

    This class defines an API endpoint for handling HTTP POST requests to predict assignments.
    It uses the 'predict_assignment' function to generate predictions 
    and 'build_result' to format the result.

    Methods:
        - post(request, format=None): Handles POST requests to the endpoint.

    Example:
        >>> # Make a POST request to predict assignments
        >>> # and receive a formatted result
        >>> response = AssignmentPrediction().post(request_data)

    Note:
        - Ensure that the 'predict_assignment' 
        and 'build_result' functions are defined and accessible.
    """
    def post(self, request, format=None):
        """
        Handles HTTP POST requests to predict assignments.

        Parameters:
            request: The HTTP request object.
            format: The requested format for the response.

        Returns:
            Response: The HTTP response containing the prediction result.

        Example:
            >>> response = AssignmentPrediction().post(request_data)

        Note:
            - If successful, the response includes a status code, 
            a success message, and the prediction result.
            - If an error occurs, the response includes 
            a status code, an error message, and no result.
        """
        try:
            result = predict_assignment(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)


class CampaignPrediction(APIView):
    """
    API endpoint for predicting campaigns.

    This class defines an API endpoint for handling HTTP POST requests to predict campaigns.
    It uses the 'predict_campaign' function to generate predictions 
    and 'build_result' to format the result.

    Methods:
        - post(request, format=None): Handles POST requests to the endpoint.

    Example:
        >>> # Make a POST request to predict campaigns
        >>> # and receive a formatted result
        >>> response = CampaignPrediction().post(request_data)

    Note:
        - Ensure that the 'predict_campaign' and 'build_result' 
        functions are defined and accessible.
    """
    def post(self, request, format=None):
        """
        Handles HTTP POST requests to predict campaigns.

        Parameters:
            request: The HTTP request object.
            format: The requested format for the response.

        Returns:
            Response: The HTTP response containing the prediction result.

        Example:
            >>> response = CampaignPrediction().post(request_data)

        Note:
            - If successful, the response includes a status code, 
            a success message, and the prediction result.
            - If an error occurs, the response includes 
            a status code, an error message, and no result.
        """
        try:
            result = predict_campaign(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)


def build_result(result):
    """
    Build a standardized result structure for API responses.

    Parameters:
        result: The result data to be included in the response.

    Returns:
        dict: A dictionary containing a standardized structure for API responses.

    Example:
        >>> result_data = {'collector_name': 'John Doe'}
        >>> response = build_result(result_data)

    Note:
        - The response structure includes a status code, a message, and the actual result.
    """
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }
    return result
