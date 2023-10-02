from .apps import Fitur7Config
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_7.libraries.utils import transform_workload_pred_input, transform_workload_pred_output

class WorkloadPrediction(APIView):
    def post(self, request, format=None):
        try:
            data =  transform_workload_pred_input(request.data)
            model = Fitur7Config.workload_pred_model
            prediction = model.predict(data)
            result = transform_workload_pred_output(prediction)
            return Response(result, status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

def test():
    return None