import asyncio
import numpy as np
from typing import List
from PIL import Image

from .common import CommonOCR
from ..config import OcrConfig
from ..utils import Quadrilateral
from ..utils.log import get_logger

logger = get_logger(__name__)

_MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'


class QwenVLOCR(CommonOCR):
    """Scene-text OCR using Qwen2.5-VL-2B-Instruct.
    State-of-the-art vision-language model — handles varied fonts, scripts,
    and fragmented manga/manhwa text without task-specific fine-tuning."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import os, torch
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            ModelClass = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            ModelClass = Qwen2VLForConditionalGeneration

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if self._device == 'cuda' else torch.float32
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

        logger.info(f'[OCR] QwenVL: betöltés ({_MODEL_ID}) → {self._device} {dtype}')
        if not token:
            logger.warning(
                '[OCR] QwenVL: HF_TOKEN nincs beállítva. Ha 401 hibát kapsz, futtasd: '
                'huggingface-cli login  VAGY  set HF_TOKEN=<token>'
            )
        try:
            self._model = ModelClass.from_pretrained(
                _MODEL_ID,
                torch_dtype=dtype,
                device_map='auto',
                token=token,
            )
        except OSError as e:
            raise RuntimeError(
                f'[OCR] QwenVL: nem sikerült betölteni a modellt ({_MODEL_ID}).\n'
                f'1. Fogadd el a licencet: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct\n'
                f'2. Jelentkezz be: huggingface-cli login\n'
                f'Eredeti hiba: {e}'
            ) from e
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            _MODEL_ID,
            min_pixels=28 * 28,
            max_pixels=512 * 28 * 28,
            token=token,
        )
        logger.info('[OCR] QwenVL: modell betöltve')

    def _read_crop(self, crop_rgb: np.ndarray) -> tuple:
        """Synchronous inference on a single RGB crop. Returns (text, prob)."""
        import torch

        h, w = crop_rgb.shape[:2]
        # Upscale tiny crops so the VL model gets enough pixels
        if h < 32 or w < 32:
            scale = max(32 / h, 32 / w, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            import cv2
            crop_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        pil_img = Image.fromarray(crop_rgb)

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': pil_img},
                {'type': 'text',
                 'text': 'Read the text in this image exactly as written. '
                         'Output only the text, nothing else.'},
            ],
        }]

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=[pil_img],
            return_tensors='pt',
        ).to(self._device)

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.1,
            )

        generated = out_ids[:, inputs['input_ids'].shape[1]:]
        text = self._processor.batch_decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return text, 0.92 if text else 0.0

    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral],
                         config: OcrConfig, verbose: bool = False) -> List[Quadrilateral]:
        self._ensure_loaded()
        loop = asyncio.get_event_loop()

        for tl in textlines:
            try:
                x1, y1, x2, y2 = tl.xyxy
                x1 = max(0, int(x1) - 4)
                y1 = max(0, int(y1) - 4)
                x2 = min(image.shape[1], int(x2) + 4)
                y2 = min(image.shape[0], int(y2) + 4)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2].copy()
                tl.text, tl.prob = await loop.run_in_executor(None, self._read_crop, crop)
                if verbose and tl.text:
                    logger.info(f'[OCR] QwenVL: {repr(tl.text)} (prob={tl.prob:.2f})')
            except Exception as e:
                logger.warning(f'[OCR] QwenVL hiba: {e}')
                tl.text = ''
                tl.prob = 0.0

        return textlines
