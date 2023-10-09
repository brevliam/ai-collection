from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from fitur_2.function.feature import predict_debtor_label_by_age, predict_debtor_label_by_location, predict_debtor_label_by_behavior
from fitur_2.function.feature import predict_debtor_label_by_character, predict_debtor_label_by_collector_field, predict_debtor_label_by_SES, predict_debtor_label_by_demography
from fitur_2.function.feature import predict_collector_label_by_age, predict_collector_label_by_location, predict_collector_label_by_behavior
from fitur_2.function.feature import predict_collector_label_by_character, predict_collector_label_by_collector_field, predict_collector_label_by_SES, predict_collector_label_by_demography

class DebtorLabelByAge(APIView):
    def post(self, request, format=None):
        try:
            result = predict_debtor_label_by_age(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class DebtorLabelByLocation(APIView):
    def post(self, request, format=None):
        try:
            result = predict_debtor_label_by_location(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class DebtorLabelByBehavior(APIView):
    def post(self, request, format=None):
        try:
            result = predict_debtor_label_by_behavior(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class DebtorLabelByCharacter(APIView):
    def post(self, request, format=None):
        try:
            result = predict_debtor_label_by_character(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class DebtorLabelByCollectorField(APIView):
    def post(self, request, format=None):
        try:
            result = predict_debtor_label_by_collector_field(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class DebtorLabelBySES(APIView):
    def post(self, request, format=None):
        try:
            result = predict_debtor_label_by_SES(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class DebtorLabelByDemography(APIView):
    def post(self, request, format=None):
        try:
            result = predict_debtor_label_by_demography(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class CollectorLabelByAge(APIView):
    def post(self, request, format=None):
        try:
            result = predict_collector_label_by_age(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class CollectorLabelByLocation(APIView):
    def post(self, request, format=None):
        try:
            result = predict_collector_label_by_location(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class CollectorLabelByBehavior(APIView):
    def post(self, request, format=None):
        try:
            result = predict_collector_label_by_behavior(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class CollectorLabelByCharacter(APIView):
    def post(self, request, format=None):
        try:
            result = predict_collector_label_by_character(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class CollectorLabelByCollectorField(APIView):
    def post(self, request, format=None):
        try:
            result = predict_collector_label_by_collector_field(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class CollectorLabelBySES(APIView):
    def post(self, request, format=None):
        try:
            result = predict_collector_label_by_SES(request.data)
            return Response(build_result(result), status = status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e) 
            return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
        
class CollectorLabelByDemography(APIView):
    def post(self, request, format=None):
        try:
            result = predict_collector_label_by_demography(request.data)
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