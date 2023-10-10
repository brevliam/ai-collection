from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_18.function.feature import Prediction#, credit_risk ,time_to_collect, total_cost

class loss_reverse(APIView):
    def post(self, request, format=None):
        try: 
            result = Prediction()
            result = result.loss_reverse(request)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        

class time_to_collect(APIView):
    def post(self, request, format=None):
        try:
            result = Prediction()
            result = result.time_to_collect(request)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

class total_cost(APIView):
    def post(self, request, format=None):
        try:
            result = Prediction()
            result = result.total_cost(request)
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