from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import strokes_to_image, segment_characters_from_image
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.decorators import api_view, parser_classes
from django.shortcuts import render
from django.http import JsonResponse
import torch
@api_view(['POST'])
@parser_classes([JSONParser, FormParser, MultiPartParser])
def recognize_strokes(request):
    """
    POST JSON:
    {
        "strokes": [
            [[x1,y1],[x2,y2],...],  # stroke 1
            [[x1,y1],[x2,y2],...]   # stroke 2
        ]
    }
    """
    try:
        strokes = request.data.get('strokes', [])
        if not strokes:
            return Response({"error": "No strokes provided"}, status=400)

        img = strokes_to_image(strokes)
        img.save("debug.png")  # Lưu file debug.png trong thư mục project
        chars = segment_characters_from_image(img, k=5)  # [(box, predictions)]

        results = []
        for box, preds in chars:
            results.append({
                "box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]},
                "top5": [{"label": l, "prob": p} for l, p in preds]
            })

        return Response({"num_chars": len(results), "predictions": results})

    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['GET'])
def draw_view(request):
    return render(request, 'draw.html')


@api_view(['GET'])
def check_device(request):
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return JsonResponse({"cuda_available": True, "device_name": device_name})
    else:
        return JsonResponse({"cuda_available": False, "device_name": "CPU"})
