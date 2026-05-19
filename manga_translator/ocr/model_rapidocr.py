import os
import cv2
import numpy as np
from typing import List

from .common import CommonOCR
from ..config import OcrConfig
from ..utils import Quadrilateral
from ..utils.log import get_logger

logger = get_logger(__name__)


def _add_torch_lib_to_dll_path():
    """Add PyTorch's lib directory to Windows DLL search path.
    onnxruntime-gpu needs cudnn64_9.dll which PyTorch bundles but doesn't expose globally.
    Must run before onnxruntime is first imported — called at module level."""
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib)
    except Exception:
        pass


# Register the DLL path at import time so it is always set before any code
# (e.g. booru_tagger.py) imports onnxruntime and its provider list is locked in.
_add_torch_lib_to_dll_path()


class RapidOCRModel(CommonOCR):
    """Scene-text OCR using RapidOCR (ONNX runtime).
    Handles game UI fonts, bold ALL-CAPS text on colored backgrounds —
    fonts that manga-specific models cannot read."""

    def __init__(self):
        self._engine = None

    def _ensure_loaded(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                use_cuda = 'CUDAExecutionProvider' in providers
                logger.info(f'[OCR] RapidOCR: elérhető providerek: {providers}')
            except Exception:
                use_cuda = False

            if use_cuda:
                try:
                    self._engine = RapidOCR(
                        det_use_cuda=True,
                        cls_use_cuda=True,
                        rec_use_cuda=True,
                    )
                    logger.info('[OCR] RapidOCR: GPU (CUDA) mode aktív')
                except Exception as e:
                    logger.warning(f'[OCR] RapidOCR: GPU init sikertelen ({e}), CPU fallbackre vált')
                    self._engine = RapidOCR()
            else:
                logger.info('[OCR] RapidOCR: CPU mode (CUDA nem elérhető)')
                self._engine = RapidOCR()

    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral],
                         config: OcrConfig, verbose: bool = False) -> List[Quadrilateral]:
        self._ensure_loaded()
        for tl in textlines:
            try:
                x1, y1, x2, y2 = tl.xyxy
                x1 = max(0, int(x1) - 2)
                y1 = max(0, int(y1) - 2)
                x2 = min(image.shape[1], int(x2) + 2)
                y2 = min(image.shape[0], int(y2) + 2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                # Crop is already a tight textline region — skip detection and
                # classification, run only the recognizer (~5x faster).
                result, _ = self._engine(crop_bgr, use_det=False, use_cls=False)
                if not result:
                    result, _ = self._engine(crop_bgr)
                if result:
                    lines = [r[1].strip() for r in result if r[2] > 0.3 and r[1].strip()]
                    tl.text = '\n'.join(lines)
                    tl.prob = max(r[2] for r in result)
                else:
                    tl.text = ''
                    tl.prob = 0.0
            except Exception:
                tl.text = ''
                tl.prob = 0.0
        return textlines
