"""
This module contains the view for predicting recommended credit.
"""


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_9.function.feature import predict_recommended_solution

class KreditRecommendedPrediction(APIView):
    """
    API view for predicting recommended credit.

    Attributes:
        None

    Methods:
        post(request, format=None): Predict recommended credit based on the input data.

    """

    def post(self, request ):
        """
            request (Request): The HTTP request object.
            format (str, optional): The format of the requested response.

        Returns:
            Response: The HTTP response object.

        """
        try:
            result = predict_recommended_solution(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

def build_result(result):
    """
    Build the result dictionary.

    Args:
        result (dict): The result data.

    Returns:
        dict: The built result dictionary.

    """
    result_dict = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result_dict
