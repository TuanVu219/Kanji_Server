from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import strokes_to_image, segment_characters_from_image
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.decorators import api_view, parser_classes
from django.shortcuts import render
from django.http import JsonResponse
import torch
import json
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import json
import requests
from django.conf import settings
from rest_framework.decorators import api_view, parser_classes,renderer_classes
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer # <--- 2. Import JSONRenderer chuẩn (quan trọng)
# 🔑 API KEY Gemini
GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_MODEL = "models/gemini-flash-lite-latest"  # GIỮ NGUYÊN
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent"
@api_view(["POST"])
@renderer_classes([JSONRenderer])
@parser_classes([JSONParser])
def gemini_translate(request):
    """
    API nhận text OCR, tách dòng, dịch thuật ngữ nghĩa (Literal Translation)
    để giữ nguyên cấu trúc dòng, và in log chi tiết ra console.
    """
    ocr_text = request.data.get("text", "").strip()

    if not ocr_text:
        return Response({"error": "Empty text"}, status=400)

    # ===== 1. TÁCH DÒNG & XỬ LÝ ĐẦU VÀO =====
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    line_count = len(lines)

    # >>>>> [DEBUG] IN INPUT RA CONSOLE <<<<<
    print(f"\n{'='*20} DEBUG OCR INPUT (Gửi đi) {'='*20}")
    print(f"🔹 Tổng số dòng: {line_count}")
    print(json.dumps(lines, indent=2, ensure_ascii=False)) 
    print(f"{'='*20} END DEBUG INPUT {'='*20}\n")
    # >>>>> [END DEBUG] <<<<<

    if not lines:
        return Response({"error": "No valid OCR lines found"}, status=400)

    # ===== 2. TẠO PROMPT "LITERAL" (CHỐNG GỘP DÒNG) =====
    prompt_text = f"""

    NHIỆM VỤ:
    - Input: Mảng JSON chứa {line_count} dòng tiếng Nhật.
    - Output: Mảng JSON chứa {line_count} dòng tiếng Việt.

    QUY TẮC BẮT BUỘC (CRITICAL):
    1. Dịch sang tiếng Việt một cách tự nhiên nhất và dễ hiểu nhất, không dịch word by word hoặc quá cứng
    2. GIỮ NGUYÊN CẤU TRÚC NGẮT DÒNG CỦA ẢNH GỐC.
    
   

    Input Data: 
    {json.dumps(lines, ensure_ascii=False)}

    Output Schema:
    {{
      "data": [
        {{ "src": "dòng gốc 1", "dst": "dịch dòng 1" }},
        {{ "src": "dòng gốc 2", "dst": "dịch dòng 2" }}
      ]
    }}
    """

    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }],
        "generationConfig": {
            "temperature": 0.1, # Nhiệt độ thấp để AI tập trung vào quy tắc
            "responseMimeType": "application/json"
        }
    }

    try:
        # ===== 3. GỌI API GEMINI =====
        res = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=30
        )

        if res.status_code != 200:
            return Response({
                "error": "Gemini API Error", 
                "status": res.status_code, 
                "detail": res.text
            }, status=res.status_code)

        data_res = res.json()
        
        # Lấy text thô và làm sạch Markdown
        raw_content = data_res.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        clean_json_str = raw_content.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        try:
            parsed_data = json.loads(clean_json_str)
            translated_items = parsed_data.get("data", [])
        except json.JSONDecodeError:
            print(f"❌ JSON ERROR RAW: {raw_content}") 
            return Response({"error": "AI response format invalid", "raw": raw_content}, status=500)

        # ===== 4. GHÉP DỮ LIỆU & LOG KẾT QUẢ =====
        final_jp = []
        final_vi = []
        debug_output = [] # Mảng dùng để in log

        # Duyệt theo mảng gốc để đảm bảo an toàn
        for i in range(line_count):
            original_line = lines[i]
            
            if i < len(translated_items):
                translated_text = translated_items[i].get("dst", "...")
            else:
                translated_text = "..." # Placeholder nếu AI trả thiếu

            final_jp.append(original_line)
            final_vi.append(translated_text)
            
            # Thêm vào mảng debug
            debug_output.append({
                "🇯🇵 JP": original_line,
                "🇻🇳 VI": translated_text
            })

        # >>>>> [DEBUG] IN OUTPUT RA CONSOLE <<<<<
        print(f"\n{'='*20} DEBUG RESULT (Kết quả nhận về) {'='*20}")
        # In từng cặp đối chiếu để dễ kiểm tra xem có bị lệch dòng không
        print(json.dumps(debug_output, indent=2, ensure_ascii=False))
        print(f"{'='*20} END DEBUG RESULT {'='*20}\n")
        # >>>>> [END DEBUG] <<<<<

        # ===== 5. TRẢ VỀ RESPONSE CHO CLIENT =====
        return Response({
            "japanese": "\n".join(final_jp),
            "vietnamese": "\n".join(final_vi),
            "line_count": line_count
        })

    except requests.exceptions.Timeout:
        return Response({"error": "Request timed out"}, status=504)
    except Exception as e:
        print(f"❌ SERVER ERROR: {str(e)}")
        return Response({"error": str(e)}, status=500)
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
