"""
Module containing an APIView for predicting collection difficulty score.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from fitur_1.function.feature import process_and_predict
from fitur_1.serializers import PrediksiSerializer
from .libraries.utils import save_input_data_to_csv

class PredictDifficultyScore(APIView):
    """
    APIView for predicting collection difficulty score.
    """

    def post(self, request):
        """
        Handle POST requests for predicting collection difficulty score.

        Parameters:
        - request (rest_framework.request.Request): The HTTP request object.

        Returns:
        Response: HTTP response containing the predicted score and category.
        """
        try:
            serializer = PrediksiSerializer(data=request.data)
            if serializer.is_valid():
                input_data = serializer.validated_data

                # Process input data
                _, score, category = process_and_predict(input_data)

                # Save input data to CSV without dropping unwanted columns
                save_input_data_to_csv(input_data, score, category)

                # Prepare the response data
                response_data = {
                    'collection_difficulty_score': round(score),
                    'collection_difficulty_category': category,
                }

                result = self.build_result(response_data)
                return Response(result)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

    def build_result(self, result):
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
