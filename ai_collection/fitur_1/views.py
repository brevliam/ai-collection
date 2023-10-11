from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .libraries.utils import save_input_data_to_csv
from fitur_1.function.feature import process_and_predict
from .serializers import PrediksiSerializer, PrediksiResponseSerializer

class PredictDifficultyScore(APIView):
    def post(self, request):
        serializer = PrediksiSerializer(data=request.data)
        if serializer.is_valid():
            input_data = serializer.validated_data

            # Proses input data
            input_array, score, category = process_and_predict(input_data)

            # Simpan input data ke CSV tanpa menjatuhkan kolom yang tidak diinginkan
            save_input_data_to_csv(input_data, score, category)

            # Prepare the response data
            response_data = {
                'collection_difficulty_score': score,
                'collection_difficulty_category': category,
            }

            result = self.build_result(response_data)
            return Response(result)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def build_result(self, result):
        result = {
            "status": 200,
            "message": "success",
            "result": result
        }
        return result
