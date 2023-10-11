from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from fitur_2.function.feature import predict_debtor_label
from fitur_2.function.feature import predict_collector_label

class DebtorLabel(APIView):
    def post(self, request, format=None):
        try:
            combined_results = predict_debtor_label(request.data)
            return Response(build_result(combined_results.to_dict(orient='records')), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class CollectorLabel(APIView):
    def post(self, request, format=None):
        try:
            combined_results = predict_collector_label(request.data)
            return Response(build_result(combined_results.to_dict(orient='records')), status=status.HTTP_200_OK)
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