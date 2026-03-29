import numpy as np
import torch
from PIL import Image as PILImage

from ..utils import InfererModule


BUBBLE_CLASS_LABELS = {0: 'bubble', 1: 'text_bubble', 2: 'text_free'}
MODEL_REPO = 'ogkalu/comic-text-and-bubble-detector'


class ComicBubbleHFDetector(InfererModule):
    """
    Wraps the ogkalu/comic-text-and-bubble-detector RT-DETR-v2 model from HuggingFace.
    Detects three region types in comic/manga images:
      0 - bubble     : speech bubble outline
      1 - text_bubble: text inside a speech bubble
      2 - text_free  : text outside any speech bubble
    Used as a pre-filter before the main text detector to restrict detection
    to regions that actually contain text or speech bubbles.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.device = 'cpu'

    async def load(self, device: str):
        if self.model is not None:
            return
        try:
            from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
        except ImportError:
            raise ImportError(
                'transformers>=4.49.0 is required for the bubble pre-filter. '
                'Install it with: pip install "transformers>=4.49.0"'
            )

        self.logger.info(f'Loading comic bubble detector from {MODEL_REPO} ...')
        self.processor = RTDetrImageProcessor.from_pretrained(MODEL_REPO)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_REPO)
        self.model.eval()
        self.device = device
        if device in ('cuda', 'mps'):
            self.model = self.model.to(device)
        self.logger.info(f'Comic bubble detector loaded on {device}')

    def detect_regions(self, image: np.ndarray, confidence: float = 0.3):
        """
        Run inference on a numpy RGB image.

        Returns a list of (box, label_id) where:
          box      - np.ndarray [x1, y1, x2, y2] in pixel coordinates
          label_id - int (0=bubble, 1=text_bubble, 2=text_free)
        """
        pil_img = PILImage.fromarray(image)
        inputs = self.processor(images=pil_img, return_tensors='pt')
        if self.device in ('cuda', 'mps'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        h, w = image.shape[:2]
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[h, w]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=confidence, target_sizes=target_sizes
        )[0]

        boxes = results['boxes'].cpu().numpy()   # (N, 4) xyxy
        labels = results['labels'].cpu().numpy() # (N,)

        return [(box, int(label)) for box, label in zip(boxes, labels)]


_bubble_detector_instance: ComicBubbleHFDetector = None


def get_bubble_detector() -> ComicBubbleHFDetector:
    global _bubble_detector_instance
    if _bubble_detector_instance is None:
        _bubble_detector_instance = ComicBubbleHFDetector()
    return _bubble_detector_instance
