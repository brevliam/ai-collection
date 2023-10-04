# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from .serializers import PrediksiSerializer, PrediksiResponseSerializer
# from fitur_1.function.feature import process_input_data, determine_score_category
# from .libraries.utils import save_input_data_to_csv
# from django.views.decorators.csrf import csrf_exempt

# @api_view(['POST'])
# @csrf_exempt
# def prediksi_view(request):
#     if request.method == 'POST':
#         serializer = PrediksiSerializer(data=request.data)
#         if serializer.is_valid():
#             input_data = serializer.validated_data

#             # Perform prediction using your model
#             input_array = process_input_data(input_data)
#             collection_difficulty_score = model.predict(input_array)[0]

#             # Determine the prediction result based on criteria
#             collection_difficulty_category = determine_score_category(collection_difficulty_score)
            
#             # Save input data to CSV without dropping unwanted fields
#             save_input_data_to_csv(input_data, collection_difficulty_score, collection_difficulty_category)

#             # Prepare the response data
#             response_data = {
#                 'collection_difficulty_score': collection_difficulty_score,
#                 'collection_difficulty_category': collection_difficulty_category,
#             }

#             # Serialize the response data
#             response_serializer = PrediksiResponseSerializer(data=response_data)

#             if response_serializer.is_valid():
#                 return Response(response_serializer.data)
#             else:
#                 return Response(response_serializer.errors, status=400)
#         return Response(serializer.errors, status=400)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from .libraries.utils import save_input_data_to_csv
from django.views.decorators.csrf import csrf_exempt
from fitur_1.function.feature import process_and_predict

@api_view(['POST'])
@csrf_exempt
def predict_difficulty_score(request):
    if request.method == 'POST':
        input_data = request.data

        # Proses input data
        input_array, score, category = process_and_predict(input_data)

        # Simpan input data ke CSV tanpa menjatuhkan kolom yang tidak diinginkan
        save_input_data_to_csv(input_data, score, category)

        # Prepare the response data
        response_data = {
            'collection_difficulty_score': score,
            'collection_difficulty_category': category,
        }

        result = build_result(response_data)
        return Response(result)

def build_result(result):
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result
