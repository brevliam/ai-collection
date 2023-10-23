from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_16.function.feature import predict_fraud_category, predict_appraisal_score

class AuctionfraudPrediction(APIView):
    def post(self, request, format=None):
        try:
            result = predict_fraud_category(request.data)

            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class CollateralAppraisal(APIView):
    def post(self, request, format=None):
        try:
            result = predict_appraisal_score(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

def build_result(result):
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result