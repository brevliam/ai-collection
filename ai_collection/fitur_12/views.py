from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from fitur_12.function.feature import predict_tenor, predict_loan

# Create your views here.
@api_view(["POST"])
def recomendation_tenor(request):
	try:
		mydata = request.data
		result = predict_tenor(mydata)
		return Response(build_result(result), status = status.HTTP_200_OK)
	except ValueError as e:
		return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
	
@api_view(["POST"])
def request_loan(request):
	try:
		mydata = request.data
		result = predict_loan(mydata)
		return Response(build_result(result), status = status.HTTP_200_OK)
	except ValueError as e:
		return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

def build_result(result):
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }
    return result