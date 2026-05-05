"""
Manga context lookup via AniList GraphQL API.
Results are cached in manga_cache.json so AniList is only queried once per title.
The current context is stored in a module-level variable so any translator can
read it without needing to thread it through every call site.
"""
import os
import re
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger('manga-translator.manga_context')

ANILIST_API = 'https://graphql.anilist.co'
CACHE_FILE = Path(__file__).parent.parent / 'manga_cache.json'

_QUERY = """
query ($search: String) {
  Media(search: $search, type: MANGA) {
    title { romaji english native }
    genres
    description(asHtml: false)
    countryOfOrigin
  }
}
"""

# Module-level: the context string for the manga currently being translated.
# Set by local.py before each new manga folder; read by chatgpt.py.
_current_manga_context: Optional[str] = None


def set_current_context(ctx: Optional[str]) -> None:
    global _current_manga_context
    _current_manga_context = ctx


def get_current_context() -> Optional[str]:
    return _current_manga_context


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8'
        )
    except Exception as e:
        logger.warning(f'[MangaContext] Cache mentési hiba: {e}')


# ---------------------------------------------------------------------------
# Title cleaning
# ---------------------------------------------------------------------------

def clean_folder_title(raw: str) -> str:
    """Remove leading digits/punctuation from folder names like '1The Tutorial Is Too Hard'."""
    return re.sub(r'^[\d\s.\-_]+', '', raw).strip()


# ---------------------------------------------------------------------------
# AniList query
# ---------------------------------------------------------------------------

async def _query_anilist(search: str) -> Optional[dict]:
    try:
        import aiohttp
    except ImportError:
        logger.debug('[MangaContext] aiohttp nincs telepítve, kihagyva.')
        return None

    token = os.getenv('ANILIST_TOKEN', '').strip()
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    payload = {'query': _QUERY, 'variables': {'search': search}}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                ANILIST_API,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return (data.get('data') or {}).get('Media')
                logger.debug(f'[MangaContext] AniList HTTP {resp.status}')
    except Exception as e:
        logger.debug(f'[MangaContext] AniList hiba: {e}')
    return None


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_context(media: dict) -> str:
    titles = media.get('title') or {}
    title = titles.get('english') or titles.get('romaji') or ''
    genres = ', '.join((media.get('genres') or [])[:5])
    country = media.get('countryOfOrigin', '')
    origin = {'KR': 'Korean manhwa', 'CN': 'Chinese manhua', 'JP': 'Japanese manga'}.get(
        country, 'manga'
    )

    desc = re.sub(r'<[^>]+>', '', media.get('description') or '').strip()
    if len(desc) > 250:
        desc = desc[:250].rsplit(' ', 1)[0] + '...'

    parts = [f'Title: "{title}" ({origin})']
    if genres:
        parts.append(f'Genres: {genres}')
    if desc:
        parts.append(f'Synopsis: {desc}')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_manga_context(folder_title: str) -> Optional[str]:
    """
    Given a raw manga folder name, return a context string for the system prompt.
    Always safe to call — returns None on any error or if not found.
    Results are cached in manga_cache.json.
    """
    title = clean_folder_title(folder_title)
    if not title:
        return None

    cache = _load_cache()
    cache_key = title.lower()

    if cache_key in cache:
        entry = cache[cache_key]
        if entry is None:
            return None  # Korábban nem találtuk
        return entry.get('context')

    logger.info(f'[MangaContext] AniList lekérdezés: "{title}"')
    media = await _query_anilist(title)

    if media:
        ctx_str = _format_context(media)
        cache[cache_key] = {'context': ctx_str}
        _save_cache(cache)
        logger.info(f'[MangaContext] Megtalálva és elmentve: "{title}"')
        return ctx_str
    else:
        cache[cache_key] = None  # Cache a miss-t is, hogy ne kérdezze le újra
        _save_cache(cache)
        logger.info(f'[MangaContext] Nem található AniList-en: "{title}"')
        return None
