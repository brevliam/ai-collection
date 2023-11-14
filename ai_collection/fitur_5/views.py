"""
Module untuk melakukan prediksi terkait fitur-fitur tertentu.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from fitur_5.function.feature import (
    predict_best_time_to_bill,
    predict_recommended_collectors_assignments,
    predict_interaction_efficiency,
)

def build_result(result):
    """
    Membangun hasil respons untuk dikirimkan sebagai respons API.

    Parameters:
    result (any): Hasil prediksi atau informasi lainnya.

    Returns:
    dict: Dictionary yang berisi informasi respons.
    """
    message = {
        "status": 200,
        "message": "success",
        "result": result,
    }
    return message

class BestTimetoBillPrediction(APIView):
    """
    Kelas untuk menangani prediksi waktu terbaik untuk melakukan penagihan.
    """
    def post(self, request):
        """
        Metode untuk menangani permintaan POST terkait prediksi waktu terbaik untuk penagihan.

        Parameters:
        - request (rest_framework.request.Request): Permintaan HTTP.
        - custom_format (str, optional): Format respons yang diinginkan.

        Returns:
        rest_framework.response.Response: Respons HTTP.
        """
        try:
            result = predict_best_time_to_bill(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = {
                "error": str(e),
                "status": status.HTTP_400_BAD_REQUEST,
            }
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)

class RecSysCollectorAssignmentsPrediction(APIView):
    """
    Kelas untuk menangani prediksi penugasan kolektor yang direkomendasikan.
    """
    def post(self, request):
        """
        Metode untuk menangani permintaan POST terkait prediksi penugasan 
        kolektor yang direkomendasikan.

        Parameters:
        - request (rest_framework.request.Request): Permintaan HTTP.
        - custom_format (str, optional): Format respons yang diinginkan.

        Returns:
        rest_framework.response.Response: Respons HTTP.
        """
        try:
            result = predict_recommended_collectors_assignments(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = {
                "error": str(e),
                "status": status.HTTP_400_BAD_REQUEST,
            }
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)

class InteractionEfficiencyClusterPrediction(APIView):
    """
    Kelas untuk menangani prediksi efisiensi interaksi dalam klaster.
    """
    def post(self, request):
        """
        Metode untuk menangani permintaan POST terkait prediksi efisiensi interaksi dalam klaster.

        Parameters:
        - request (rest_framework.request.Request): Permintaan HTTP.
        - custom_format (str, optional): Format respons yang diinginkan.

        Returns:
        rest_framework.response.Response: Respons HTTP.
        """
        try:
            result = predict_interaction_efficiency(request.data)
            return Response(build_result(result), status=status.HTTP_200_OK)
        except Exception as e:
            error_message = {
                "error": str(e),
                "status": status.HTTP_400_BAD_REQUEST,
            }
            return Response(error_message, status=status.HTTP_400_BAD_REQUEST)
