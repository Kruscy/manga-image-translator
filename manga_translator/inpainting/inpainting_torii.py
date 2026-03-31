import base64

import aiohttp
import cv2
import numpy as np

from .common import CommonInpainter
from ..config import InpainterConfig


TORII_INPAINT_URL = 'https://api.toriitranslate.com/api/inpaint'


class ToriiInpainter(CommonInpainter):
    """
    Inpainting using the ToriiTranslate API (/api/inpaint).
    Sends the image and mask; white areas in the mask are inpainted.
    Costs 0.02 credits per request.
    Requires TORII_API_KEY in the .env file.
    """

    async def _inpaint(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        from ..translators.keys import TORII_API_KEY
        if not TORII_API_KEY:
            raise ValueError('TORII_API_KEY is not set. Add TORII_API_KEY=<your_key> to your .env file.')

        # Encode image as PNG bytes
        _, img_encoded = cv2.imencode('.png', image)
        img_bytes = img_encoded.tobytes()

        # Normalize mask to single-channel uint8 (white = inpaint)
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask.copy()
        mask_binary = np.where(mask_gray > 0, 255, 0).astype(np.uint8)

        _, mask_encoded = cv2.imencode('.png', mask_binary)
        mask_bytes = mask_encoded.tobytes()

        headers = {'Authorization': f'Bearer {TORII_API_KEY}'}

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field('image', img_bytes, filename='image.png', content_type='image/png')
            form.add_field('mask', mask_bytes, filename='mask.png', content_type='image/png')

            async with session.post(TORII_INPAINT_URL, headers=headers, data=form) as response:
                if response.headers.get('success') == 'true':
                    data = await response.json(content_type=None)
                    image_b64 = data['image'].split(',')[1]
                    img_array = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
                    result = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if result is None:
                        self.logger.warning('Torii inpainting returned an undecodable image, using original.')
                        return image
                    return result
                else:
                    content = await response.text()
                    self.logger.warning(f'Torii inpainting request failed: {content}. Using original image.')
                    return image
