"""
LaMa inpainter that uses an external lama-cleaner / iopaint server.
Falls back to lama_large if the server is unavailable or errors.
"""

import base64
import io
import numpy as np
from PIL import Image
import requests

from .common import CommonInpainter


class LaMa2Inpainter(CommonInpainter):

    _LAMA_V1_URL  = 'http://localhost:8080/api/v1/inpaint'   # iopaint (newer)
    _LAMA_OLD_URL = 'http://localhost:8080/inpaint'           # original lama-cleaner

    def __init__(self):
        super().__init__()
        self._api_url = self._detect_api()
        self._use_fallback = self._api_url is None

    # ------------------------------------------------------------------ #
    # Server detection
    # ------------------------------------------------------------------ #

    def _detect_api(self):
        """Return the working API URL, or None if server is unreachable."""
        try:
            requests.get('http://localhost:8080', timeout=3)
        except Exception:
            print('[LaMa2] Szerver nem elérhető, lama_large fallback')
            return None

        # OPTIONS on a non-existent route returns 404 without raising — must check status.
        try:
            r = requests.options(self._LAMA_V1_URL, timeout=3)
            if r.status_code != 404:
                print('[LaMa2] iopaint szerver érzékelve (v1 API)')
                return self._LAMA_V1_URL
        except Exception:
            pass

        print('[LaMa2] lama-cleaner szerver érzékelve (régi API)')
        return self._LAMA_OLD_URL

    # ------------------------------------------------------------------ #
    # Server inpainting
    # ------------------------------------------------------------------ #

    def _to_png_b64(self, arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    def _crop_for_server(self, image: np.ndarray, mask: np.ndarray, padding: int = 80):
        """Crop the masked region (+ padding), aligned to 64px, min 512×512.
        Returns (img_crop, mask_crop, x, y) where x/y is the top-left of the crop in the original."""
        h, w = image.shape[:2]
        ys, xs = np.where(mask > 128)
        if len(xs) == 0:
            return image, mask, 0, 0

        min_x = max(0, int(xs.min()) - padding)
        min_y = max(0, int(ys.min()) - padding)
        max_x = min(w - 1, int(xs.max()) + padding)
        max_y = min(h - 1, int(ys.max()) + padding)

        align = 64
        min_size = 512
        bw = max(min_size, int(np.ceil((max_x - min_x) / align)) * align)
        bh = max(min_size, int(np.ceil((max_y - min_y) / align)) * align)

        # Centre the crop over the masked region
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        x = max(0, cx - bw // 2)
        y = max(0, cy - bh // 2)
        if x + bw > w: x = max(0, w - bw)
        if y + bh > h: y = max(0, h - bh)
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        return image[y:y+bh, x:x+bw], mask[y:y+bh, x:x+bw], x, y

    async def _inpaint_server(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 2048) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]
        crop_img, crop_mask, crop_x, crop_y = self._crop_for_server(image, mask)

        img_b64  = self._to_png_b64(crop_img)
        mask_b64 = self._to_png_b64(crop_mask)

        if self._api_url == self._LAMA_V1_URL:
            # iopaint JSON API
            payload = {
                'image': img_b64,
                'mask':  mask_b64,
                'ldmSteps': 25,
                'hdStrategy': 'Crop',
                'hdStrategyCropMargin': 128,
                'hdStrategyCropTrigerSize': 800,
                'hdStrategyResizeLimit': 2048,
            }
            resp = requests.post(self._api_url, json=payload, timeout=120)
        else:
            # Original lama-cleaner multipart API — server reads ALL fields via form[key],
            # so every expected key must be present even when using LaMa (no SD/LDM).
            img_bytes  = io.BytesIO(); Image.fromarray(crop_img.astype(np.uint8)).save(img_bytes, 'PNG'); img_bytes.seek(0)
            mask_bytes = io.BytesIO(); Image.fromarray(crop_mask.astype(np.uint8)).save(mask_bytes, 'PNG'); mask_bytes.seek(0)
            resp = requests.post(
                self._api_url,
                files={'image': ('image.png', img_bytes, 'image/png'),
                       'mask':  ('mask.png',  mask_bytes, 'image/png')},
                data={
                    'ldmSteps': '25',
                    'ldmSampler': 'plms',
                    'hdStrategy': 'Crop',
                    'zitsWireframe': 'true',
                    'hdStrategyCropMargin': '128',
                    'hdStrategyCropTrigerSize': '800',
                    'hdStrategyResizeLimit': '2048',
                    'prompt': '',
                    'negativePrompt': '',
                    'useCroper': 'false',
                    'croperX': '0',
                    'croperY': '0',
                    'croperHeight': '512',
                    'croperWidth': '512',
                    'sdScale': '1.0',
                    'sdMaskBlur': '5',
                    'sdStrength': '0.75',
                    'sdSteps': '50',
                    'sdGuidanceScale': '7.5',
                    'sdSampler': 'uni_pc',
                    'sdSeed': '-1',
                    'sdMatchHistograms': 'false',
                    'cv2Flag': 'INPAINT_NS',
                    'cv2Radius': '4',
                    'paintByExampleSteps': '50',
                    'paintByExampleGuidanceScale': '7.5',
                    'paintByExampleMaskBlur': '5',
                    'paintByExampleSeed': '-1',
                    'paintByExampleMatchHistograms': 'false',
                    'p2pSteps': '50',
                    'p2pImageGuidanceScale': '1.5',
                    'p2pGuidanceScale': '7.5',
                    'controlnet_conditioning_scale': '0.4',
                    'controlnet_method': 'control_v11p_sd15_canny',
                },
                timeout=120,
            )

        if resp.status_code != 200:
            raise RuntimeError(f'Server {resp.status_code}: {resp.text[:200]}')

        # Response: either raw image bytes or JSON {"image": "base64..."}
        ct = resp.headers.get('Content-Type', '')
        if 'image' in ct:
            result = np.array(Image.open(io.BytesIO(resp.content)))
        else:
            b64 = resp.json().get('image', '')
            result = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))

        if result.ndim == 2:
            result = np.stack([result] * 3, axis=-1)
        result = result[:, :, :3]

        # Resize crop result to match crop size if server changed dimensions
        ch, cw = crop_img.shape[:2]
        if result.shape[:2] != (ch, cw):
            result = np.array(Image.fromarray(result).resize((cw, ch), Image.LANCZOS))

        # Paste crop result back into a copy of the full original image
        output = image.copy()
        output[crop_y:crop_y+ch, crop_x:crop_x+cw] = result
        return output

    # ------------------------------------------------------------------ #
    # Fallback via dispatch (avoids "model not loaded" error)
    # ------------------------------------------------------------------ #

    async def _inpaint_fallback(self, image, mask, config, inpainting_size, device, verbose):
        from . import dispatch as inpaint_dispatch
        from ..config import InpainterConfig, Inpainter
        return await inpaint_dispatch(
            Inpainter.lama_large, image, mask,
            config if config is not None else InpainterConfig(),
            inpainting_size, device, verbose,
        )

    # ------------------------------------------------------------------ #
    # CommonInpainter interface
    # ------------------------------------------------------------------ #

    async def _inpaint(self, image: np.ndarray, mask: np.ndarray,
                       config=None, inpainting_size: int = 1024,
                       verbose: bool = False) -> np.ndarray:
        if self._use_fallback:
            return await self._inpaint_fallback(image, mask, config, inpainting_size, 'cuda', verbose)

        try:
            return await self._inpaint_server(image, mask, inpainting_size)
        except Exception as e:
            print(f'[LaMa2] Szerver hiba ({e}), fallback lama_large-ra')
            self._use_fallback = True
            return await self._inpaint_fallback(image, mask, config, inpainting_size, 'cuda', verbose)

    async def inpaint(self, image: np.ndarray, mask: np.ndarray,
                      config=None, inpainting_size: int = 1024,
                      verbose: bool = False) -> np.ndarray:
        if not np.any(mask > 128):
            return image
        return await self._inpaint(image, mask, config, inpainting_size, verbose)
