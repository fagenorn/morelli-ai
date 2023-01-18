from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests

MAX_FILE_SIZE=6553500 # 6.5 MB, Changing this is not enough to change the max file size, you need to change it in the ml\serve\config.properties file as well (nginx config might also need to be changed)

@csrf_exempt
def image_upload(request):
    if request.method == 'POST':
        image = request.FILES['file']
        version = request.POST['version']

        if image.size >= MAX_FILE_SIZE:
            return JsonResponse({'label': 'FILE TOO LARGE', 'ai_chance': 1})

        api_response = requests.post(f'http://localhost:8080/predictions/morelli/{version}', data=image)
        api_response_data = api_response.json()

        if api_response.status_code == 200:
            return JsonResponse({'label': api_response_data['label'], 'ai_chance': api_response_data['ai_chance']})
        else:
            return JsonResponse({'label': 'ERROR', 'ai_chance': 1})
    else:
        return render(request, 'image_upload.html')