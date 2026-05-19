import asyncio
import os
from typing import List


class TranslationPool:
    """Translates text using a pool of OpenAI API keys in parallel."""

    def __init__(self):
        self._translators = []
        self._semaphores: List[asyncio.Semaphore] = []
        self._ready = False

    def _load_keys(self) -> List[str]:
        keys = []
        k = os.getenv('OPENAI_API_KEY', '')
        if k:
            keys.append(k)
        for i in range(2, 20):
            k = os.getenv(f'OPENAI_API_KEY_{i}', '')
            if k:
                keys.append(k)
        return keys or ['']

    def _ensure_ready(self):
        """Lazy init — called from async context so Semaphores are created in the running loop."""
        if self._ready:
            return
        from manga_translator.translators.chatgpt import OpenAITranslator
        keys = self._load_keys()
        for key in keys:
            t = OpenAITranslator(check_openai_key=False, api_key=key)
            self._translators.append(t)
            self._semaphores.append(asyncio.Semaphore(1))
        self._ready = True

    async def translate(self, texts: List[str], from_lang: str, config) -> List[str]:
        """Translate texts using any free API key from the pool.
        Blocks until a key is available, then translates and returns results."""
        self._ensure_ready()

        while True:
            for t, sem in zip(self._translators, self._semaphores):
                if sem._value > 0:
                    async with sem:
                        t.parse_args(config.translator)
                        return await t._translate(
                            from_lang,
                            config.translator.target_lang,
                            texts,
                        )
            await asyncio.sleep(0.05)


translation_pool = TranslationPool()
