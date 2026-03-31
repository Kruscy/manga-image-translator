import base64
import io
import asyncio
from typing import List

import aiohttp
import cv2
import numpy as np

from .common import CommonOCR
from ..config import OcrConfig
from ..utils import Quadrilateral


TORII_OCR_URL = 'https://api.toriitranslate.com/api/ocr'


class ToriiOCR(CommonOCR):
    """
    OCR using the ToriiTranslate API (/api/ocr).
    Sends each detected text region as a cropped image and retrieves the recognized text.
    Requires TORII_API_KEY in the .env file.
    """

    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False) -> List[Quadrilateral]:
        from ..translators.keys import TORII_API_KEY
        if not TORII_API_KEY:
            raise ValueError('TORII_API_KEY is not set. Add TORII_API_KEY=<your_key> to your .env file.')

        for textline in textlines:
            x, y, w, h = textline.aabb.x, textline.aabb.y, textline.aabb.w, textline.aabb.h
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                textline.text = ''
                textline.prob = 0.0
                continue

            _, img_encoded = cv2.imencode('.png', crop)
            img_bytes = img_encoded.tobytes()

            text, prob = await self._call_ocr(img_bytes, TORII_API_KEY)
            textline.text = text
            textline.prob = prob

        return textlines

    async def _call_ocr(self, img_bytes: bytes, api_key: str):
        headers = {'Authorization': f'Bearer {api_key}'}

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field('file', img_bytes, filename='crop.png', content_type='image/png')

            async with session.post(TORII_OCR_URL, headers=headers, data=form) as response:
                if response.headers.get('success') == 'true':
                    data = await response.json(content_type=None)
                    return self._extract_text(data)
                else:
                    content = await response.text()
                    self.logger.warning(f'Torii OCR request failed: {content}')
                    return '', 0.0

    def _extract_text(self, data):
        """
        Parse the Torii OCR JSON response.
        The response is a list of detected text objects or a single dict.
        Each object may have: text, confidence, language fields.
        """
        if isinstance(data, list):
            parts = []
            total_conf = 0.0
            for item in data:
                if isinstance(item, dict):
                    t = item.get('text', '').strip()
                    if t:
                        parts.append(t)
                        total_conf += float(item.get('confidence', 1.0))
            text = ' '.join(parts)
            prob = (total_conf / len(parts)) if parts else 0.0
            return text, min(prob, 1.0)
        elif isinstance(data, dict):
            text = data.get('text', '').strip()
            prob = float(data.get('confidence', 1.0))
            return text, min(prob, 1.0)
        return '', 0.0
