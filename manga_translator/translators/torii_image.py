"""
ToriiTranslate full-image translation.
Uses POST https://api.toriitranslate.com/api/v2/upload
Sends the full manga image; receives a fully translated image back.
This bypasses manga-image-translator's OCR, text translation, inpainting and rendering steps.
Requires TORII_API_KEY in the .env file.
"""

import base64
import io
from typing import Optional

import aiohttp
from PIL import Image

from .keys import TORII_API_KEY


TORII_TRANSLATE_URL = 'https://api.toriitranslate.com/api/v2/upload'

# Map manga-image-translator language codes to Torii ISO 639-1 codes
_LANG_MAP = {
    'ENG': 'en',
    'JPN': 'ja',
    'KOR': 'ko',
    'CHS': 'zh',
    'CHT': 'zh-TW',
    'FRA': 'fr',
    'DEU': 'de',
    'ESP': 'es',
    'PTB': 'pt',
    'RUS': 'ru',
    'ARA': 'ar',
    'HUN': 'hu',
    'NLD': 'nl',
    'ITA': 'it',
    'PLK': 'pl',
    'UKR': 'uk',
    'VIN': 'vi',
    'IND': 'id',
    'THA': 'th',
    'TRK': 'tr',
    'HRV': 'hr',
    'ROM': 'ro',
    'CSY': 'cs',
    'FIL': 'fil',
}


def _to_iso(lang_code: str) -> str:
    return _LANG_MAP.get(lang_code, lang_code.lower()[:2])


async def translate_image_with_torii(
    image: Image.Image,
    target_lang: str,
    torii_model: str = 'gemini-3-flash',
    torii_font: str = 'wildwords',
    torii_text_align: str = 'auto',
    torii_stroke_disabled: bool = False,
    torii_min_font_size: int = 12,
) -> Optional[Image.Image]:
    """
    Sends a PIL Image to Torii's v2/upload endpoint and returns the translated PIL Image.
    Returns None on failure.
    """
    api_key = TORII_API_KEY
    if not api_key:
        raise ValueError('TORII_API_KEY is not set. Add TORII_API_KEY=<your_key> to your .env file.')

    # Convert PIL Image to PNG bytes
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    img_bytes = buf.getvalue()

    iso_lang = _to_iso(target_lang)

    headers = {'Authorization': f'Bearer {api_key}'}
    data = {
        'target_lang': iso_lang,
        'translator': torii_model,
        'font': torii_font,
        'text_align': torii_text_align,
        'stroke_disabled': 'true' if torii_stroke_disabled else 'false',
        'min_font_size': str(torii_min_font_size),
    }

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        for k, v in data.items():
            form.add_field(k, v)
        form.add_field('file', img_bytes, filename='image.png', content_type='image/png')

        async with session.post(TORII_TRANSLATE_URL, headers=headers, data=form) as response:
            if response.headers.get('success') == 'true':
                resp_data = await response.json(content_type=None)
                image_b64 = resp_data['image'].split(',')[1]
                result_bytes = base64.b64decode(image_b64)
                return Image.open(io.BytesIO(result_bytes)).convert('RGB')
            else:
                content = await response.text()
                raise RuntimeError(f'Torii translation failed: {content}')
