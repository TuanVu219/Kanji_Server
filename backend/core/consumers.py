import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from PIL import Image, ImageDraw
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

from .utils import strokes_to_image, segment_characters_from_image

logger = logging.getLogger(__name__)

SERVER_CANVAS_SIZE = 600  # Kích thước canvas chuẩn của server

class RecognizeConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strokes: List[List[List[int]]] = []
        self.client_canvas_size: Optional[Dict[str, int]] = None

    async def connect(self):
        await self.accept()
        await self.send_json({"status": "connected"})
        logger.info("✅ WebSocket connected.")
    async def disconnect(self, close_code: int):
        logger.info(f"🔌 WebSocket disconnected: {close_code}")
        self.strokes.clear()
        self.client_canvas_size = None

    async def receive(self, text_data: str | None = None, bytes_data: bytes | None = None):
        if not text_data:
            return
        try:
            data = json.loads(text_data)
            action = data.get("action")

            # Cập nhật canvas_size từ client nếu có
            if isinstance(data.get("canvas_size"), dict):
                self.client_canvas_size = data.get("canvas_size")

            match action:
                case "stroke":
                    stroke = data.get("stroke")
                    if isinstance(stroke, list):
                        self.strokes.append(stroke)
                    await self._run_and_send_predictions()

                case "clear":
                    self.strokes.clear()
                    await self.send_json({"status": "cleared", "num_strokes": 0, "predictions": []})

                case "undo":
                    if self.strokes:
                        self.strokes.pop()
                    await self._run_and_send_predictions()

                case "predict_once":
                    await self._run_and_send_predictions()

                case _:
                    await self.send_json({"error": "unknown_action", "received": data})
                    logger.warning(f"⚠️ Unknown action received: {data}")

        except Exception as e:
            logger.exception("❌ Error in receive()")
            await self.send_json({"error": str(e)})

    def _compute_scale_from_strokes(self, strokes: List[List[List[int]]], server_size: int):
        """Tự động tính scale nếu client không gửi canvas_size."""
        max_x = max_y = 0
        for stroke in strokes:
            for x, y in stroke:
                if x is None or y is None:
                    continue
                if x > max_x: max_x = x
                if y > max_y: max_y = y
        max_coord = max(max_x, max_y, 1)
        if max_coord < server_size * 0.9:
            scale = server_size / float(max_coord)
            return scale, scale
        return 1.0, 1.0

    def _scale_strokes(self, strokes: List[List[List[int]]], server_size: int):
        """Scale strokes to server canvas."""
        if self.client_canvas_size and self.client_canvas_size.get("width"):
            cw = self.client_canvas_size.get("width")
            ch = self.client_canvas_size.get("height", cw)
            sx = server_size / float(cw)
            sy = server_size / float(ch)
        else:
            sx, sy = self._compute_scale_from_strokes(strokes, server_size)

        if sx == 1.0 and sy == 1.0:
            return strokes, sx, sy

        scaled = []
        for stroke in strokes:
            scaled_stroke = []
            for x, y in stroke:
                nx = int(round(x * sx))
                ny = int(round(y * sy))
                scaled_stroke.append([nx, ny])
            scaled.append(scaled_stroke)
        return scaled, sx, sy

    async def _run_and_send_predictions(self):
        if not self.strokes:
            await self.send_json({"status": "no_strokes", "num_strokes": 0, "predictions": []})
            return

        try:
            # Scale strokes
            scaled_strokes, sx, sy = self._scale_strokes(self.strokes, SERVER_CANVAS_SIZE)

            # Tạo ảnh từ strokes
            img = await sync_to_async(strokes_to_image)(scaled_strokes, SERVER_CANVAS_SIZE)

            # Gọi hàm segment + nhận dạng
            merged = await sync_to_async(segment_characters_from_image)(img, 5)


            # Chuẩn bị kết quả
            results: List[Dict[str, Any]] = []
            for (box, preds) in merged:
                bx = {"x": int(box[0]), "y": int(box[1]), "w": int(box[2]), "h": int(box[3])}
                topk = [{"label": str(l), "prob": float(p)} for l, p in preds]
                results.append({"box": bx, "topk": topk})

            payload = {
                "status": "predictions",
                "num_strokes": len(self.strokes),
                "num_chars": len(results),
                "predictions": results,
                "scale": {"sx": sx, "sy": sy}
            }

            await self.send_json(payload)
            logger.debug(f"✅ Sent predictions: {len(results)} chars, {len(self.strokes)} strokes")

        except Exception as e:
            logger.exception("❌ Error during prediction")
            await self.send_json({"error": "prediction_failed", "detail": str(e)})

    async def send_json(self, content: dict):
        await self.send(text_data=json.dumps(content, ensure_ascii=False))
