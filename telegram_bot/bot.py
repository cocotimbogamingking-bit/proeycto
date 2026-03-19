"""
Telegram Rep Search Bot — Profesional Edition
─────────────────────────────────────────────
AI-powered bot for searching replica sneakers & streetwear.
Uses Groq (Llama 3.3 / Llama 4 Scout / Whisper) + Hipopick API.
"""

import os
import re
import json
import time
import logging
import base64
import urllib.parse
import asyncio
import threading
import functools
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from difflib import SequenceMatcher

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RepBot")

# ── Configuration ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8786285615:AAGZw_Wx50V3Kdez8QG8ugJLPVfYdfxrFvk")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
AUDIO_MODEL = "whisper-large-v3"
QC_BASE = "https://d1loi7eremk1cu.cloudfront.net/"

# Limits
MAX_HISTORY_PER_USER = 20
MAX_ACTIVE_USERS = 500
RATE_LIMIT_SECONDS = 2  # Minimum seconds between search requests per user
SEARCH_CACHE_TTL = 300  # 5 minutes
MAX_RETRIES = 2
RETRY_DELAY = 1.0  # seconds

# ── Prompts ───────────────────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = (
    "Eres un asistente amigable en Telegram, experto en zapatillas y ropa streetwear (reps/réplicas). "
    "Respondes SIEMPRE en español, de forma natural, breve y con buena vibra. NO incluyas ningún tag especial. "
    "Puedes hablar de sneakers, ropa, precios, tallas, combinar outfits, o cualquier otro tema. "
    "Si el usuario pregunta cómo buscar productos, díle que simplemente lo pida natural: "
    "'quiero ver unas nike dunk' o 'buscame yeezy 350'. "
    "Nunca uses caracteres especiales de Markdown como * o _ en tu respuesta."
)

CLASSIFIER_PROMPT = (
    "Eres un clasificador de intención. Analiza el mensaje y devuelve SOLO un JSON.\n"
    "Si el usuario quiere buscar/ver/encontrar un producto: {\"intent\": \"search\", \"query\": \"término\"}\n"
    "Si es saludo, charla, pregunta u opinión: {\"intent\": \"chat\"}\n"
    "EJEMPLOS:\n"
    "- 'hola' -> {\"intent\": \"chat\"}\n"
    "- 'holaaa' -> {\"intent\": \"chat\"}\n"
    "- 'qué tal' -> {\"intent\": \"chat\"}\n"
    "- 'cómo estás' -> {\"intent\": \"chat\"}\n"
    "- 'gracias' -> {\"intent\": \"chat\"}\n"
    "- 'son buenas las reps?' -> {\"intent\": \"chat\"}\n"
    "- 'qué opinas de weidian' -> {\"intent\": \"chat\"}\n"
    "- 'quiero ver jordan 1' -> {\"intent\": \"search\", \"query\": \"jordan 1\"}\n"
    "- 'busca yeezy 350' -> {\"intent\": \"search\", \"query\": \"yeezy 350\"}\n"
    "- 'tienes reps de nike dunk' -> {\"intent\": \"search\", \"query\": \"nike dunk\"}\n"
    "- 'cuánto cuestan las air force' -> {\"intent\": \"search\", \"query\": \"air force 1\"}\n"
    "- 'me puedes buscar unas new balance 550' -> {\"intent\": \"search\", \"query\": \"new balance 550\"}\n"
    "- 'quiero unas zapatillas negras nike' -> {\"intent\": \"search\", \"query\": \"nike negro\"}\n"
    "- 'muéstrame tech fleece' -> {\"intent\": \"search\", \"query\": \"nike tech fleece\"}\n"
    "- 'busca estas zapatillas' -> {\"intent\": \"search\", \"query\": \"zapatillas\"}\n"
    "- 'tienes algo de balenciaga' -> {\"intent\": \"search\", \"query\": \"balenciaga\"}\n"
    "SOLO devuelve el JSON, nada más."
)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def escape_md(text: str) -> str:
    """Escape Markdown V1 special characters to prevent Telegram parse errors."""
    if not text:
        return ""
    # In MarkdownV1, these characters have special meaning: _ * ` [
    # We escape them so Telegram doesn't try to parse them as formatting
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


def safe_markdown(text: str) -> str:
    """Try to fix broken Markdown by removing unbalanced markers."""
    if not text:
        return ""
    # Count occurrences of markdown markers
    for marker in ("*", "_", "`"):
        count = text.count(marker)
        if count % 2 != 0:
            # Odd number = unbalanced, escape all of them
            text = text.replace(marker, f"\\{marker}")
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Simple per-user rate limiter to prevent spam."""

    def __init__(self, cooldown: float = RATE_LIMIT_SECONDS):
        self._last_request: dict[int, float] = {}
        self._cooldown = cooldown

    def check(self, user_id: int) -> bool:
        """Returns True if the user can proceed, False if rate limited."""
        now = time.time()
        last = self._last_request.get(user_id, 0)
        if now - last < self._cooldown:
            return False
        self._last_request[user_id] = now
        # Cleanup old entries periodically
        if len(self._last_request) > MAX_ACTIVE_USERS * 2:
            cutoff = now - 3600  # Remove entries older than 1 hour
            self._last_request = {
                uid: t for uid, t in self._last_request.items() if t > cutoff
            }
        return True

    def remaining(self, user_id: int) -> float:
        """Returns how many seconds until the user can make another request."""
        now = time.time()
        last = self._last_request.get(user_id, 0)
        remaining = self._cooldown - (now - last)
        return max(0, remaining)


rate_limiter = RateLimiter()


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION MEMORY (with TTL and LRU eviction)
# ══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """LRU-based conversation history with per-user message limits."""

    def __init__(self, max_users: int = MAX_ACTIVE_USERS,
                 max_messages: int = MAX_HISTORY_PER_USER):
        self._store: OrderedDict[int, list] = OrderedDict()
        self._max_users = max_users
        self._max_messages = max_messages

    def get(self, uid: int) -> list:
        """Get conversation history for a user, moving them to end (most recent)."""
        if uid in self._store:
            self._store.move_to_end(uid)
            return self._store[uid]
        return []

    def add(self, uid: int, role: str, content):
        """Add a message to user's history."""
        if uid not in self._store:
            # Evict oldest user if at capacity
            if len(self._store) >= self._max_users:
                self._store.popitem(last=False)
            self._store[uid] = []

        self._store.move_to_end(uid)
        self._store[uid].append({"role": role, "content": content})

        # Trim to max messages
        if len(self._store[uid]) > self._max_messages:
            self._store[uid] = self._store[uid][-self._max_messages:]

    def clear(self, uid: int):
        """Clear a user's conversation history."""
        self._store.pop(uid, None)


memory = ConversationMemory()


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH CACHE
# ══════════════════════════════════════════════════════════════════════════════

class SearchCache:
    """Simple TTL cache for search results to avoid redundant API calls."""

    def __init__(self, ttl: int = SEARCH_CACHE_TTL, max_entries: int = 100):
        self._cache: OrderedDict[str, tuple[float, list]] = OrderedDict()
        self._ttl = ttl
        self._max_entries = max_entries

    def _make_key(self, query: str, sort: str, top_n: int) -> str:
        return f"{query.lower().strip()}|{sort}|{top_n}"

    def get(self, query: str, sort: str = "", top_n: int = 5) -> list | None:
        """Get cached results if available and not expired."""
        key = self._make_key(query, sort, top_n)
        if key in self._cache:
            timestamp, results = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._cache.move_to_end(key)
                logger.info(f"Cache HIT for query: '{query}'")
                return results
            else:
                del self._cache[key]
        return None

    def put(self, query: str, sort: str, top_n: int, results: list):
        """Store search results in cache."""
        key = self._make_key(query, sort, top_n)
        if len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
        self._cache[key] = (time.time(), results)


search_cache = SearchCache()


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP CLIENT (singleton with connection pooling)
# ══════════════════════════════════════════════════════════════════════════════

_http_client: httpx.AsyncClient | None = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=True,
        )
    return _http_client


async def api_request_with_retry(
    method: str,
    url: str,
    retries: int = MAX_RETRIES,
    **kwargs,
) -> httpx.Response:
    """Make an HTTP request with automatic retry on transient failures."""
    client = await get_http_client()
    last_error = None

    for attempt in range(retries + 1):
        try:
            if method == "GET":
                resp = await client.get(url, **kwargs)
            elif method == "POST":
                resp = await client.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")

            resp.raise_for_status()
            return resp

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            if attempt < retries:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"Request to {url} failed (attempt {attempt + 1}/{retries + 1}): {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            else:
                logger.error(f"Request to {url} failed after {retries + 1} attempts: {e}")

        except httpx.HTTPStatusError as e:
            # Don't retry client errors (4xx), only server errors (5xx)
            if e.response.status_code >= 500 and attempt < retries:
                last_error = e
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"Server error {e.response.status_code} from {url} "
                    f"(attempt {attempt + 1}/{retries + 1}). Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            else:
                raise

    raise last_error  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
#  SMART SEARCH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

ALIASES = [
    # ══════════════════════════════════════════════════════════════
    #  NIKE — Sneakers
    # ══════════════════════════════════════════════════════════════
    (r'\baf[ -]?1s?\b',                 'air force 1'),
    (r'\bair ?force(s?)\b',             'air force 1'),
    (r'\bforces?\b',                    'air force 1'),
    (r'\bdunk(s?)\b',                   'nike dunk'),
    (r'\bdunk ?low(s?)\b',              'nike dunk low'),
    (r'\bdunk ?high(s?)\b',             'nike dunk high'),
    (r'\bpanda(s?)\b',                  'nike dunk panda'),
    (r'\bsb\b',                         'nike sb'),
    (r'\bsb ?dunk(s?)\b',              'nike sb dunk'),
    (r'\bchunky ?dunky\b',             'nike sb dunk chunky dunky'),
    (r'\bstrangelove\b',               'nike sb dunk strangelove'),
    (r'\btravis ?dunk\b',              'nike sb dunk travis scott'),
    (r'\bvapormax\b',                   'nike vapormax'),
    (r'\bvapor ?max\b',                 'nike vapormax'),
    (r'\bam[ -]?1\b',                   'nike air max 1'),
    (r'\bam[ -]?90\b',                  'nike air max 90'),
    (r'\bam[ -]?95\b',                  'nike air max 95'),
    (r'\bam[ -]?97\b',                  'nike air max 97'),
    (r'\bam[ -]?270\b',                 'nike air max 270'),
    (r'\bam[ -]?720\b',                 'nike air max 720'),
    (r'\bair ?max\b',                   'nike air max'),
    (r'\bair ?max ?plus\b',             'nike air max plus'),
    (r'\btn(s?)\b',                     'nike air max plus tn'),
    (r'\btuned\b',                      'nike air max plus tn'),
    (r'\bpresto(s?)\b',                 'nike air presto'),
    (r'\breact\b',                      'nike react'),
    (r'\bhuarache(s?)\b',              'nike huarache'),
    (r'\bcortez\b',                     'nike cortez'),
    (r'\bblaze[r]?(s?)\b',             'nike blazer'),
    (r'\bblazer ?mid\b',                'nike blazer mid'),
    (r'\bblazer ?low\b',                'nike blazer low'),
    (r'\bmonarch(s?)\b',               'nike air monarch'),
    (r'\bp[ -]?6000\b',                 'nike p-6000'),
    (r'\bv2k\b',                        'nike v2k run'),
    (r'\bzoom ?vomero\b',               'nike zoom vomero'),
    (r'\bvomero(s?)\b',                 'nike zoom vomero'),
    (r'\binitiator\b',                  'nike initiator'),
    (r'\bshox\b',                       'nike shox'),
    (r'\btailwind\b',                   'nike air tailwind'),
    (r'\bwaffle(s?)\b',                 'nike waffle'),
    (r'\bsacai\b',                      'nike sacai'),
    (r'\bld ?waffle\b',                 'nike sacai ld waffle'),
    (r'\bkobe(s?)\b',                   'nike kobe'),
    (r'\bkobe ?[456]\b',                'nike kobe'),
    (r'\blebron(s?)\b',                 'nike lebron'),
    (r'\bkyrie(s?)\b',                  'nike kyrie'),
    (r'\bkd[ -]?\d*\b',                'nike kd'),
    (r'\bpg[ -]?\d+\b',                'nike pg'),
    (r'\bfoamposite(s?)\b',            'nike foamposite'),
    (r'\bfoam(s?)\b',                   'nike foamposite'),

    # ══════════════════════════════════════════════════════════════
    #  NIKE — Ropa & Accesorios
    # ══════════════════════════════════════════════════════════════
    (r'\btech[ -]?fleece\b',            'nike tech fleece'),
    (r'\btf\b',                         'nike tech fleece'),
    (r'\btech\b',                       'nike tech fleece'),
    (r'\bwindrunner\b',                 'nike windrunner'),
    (r'\bswoosh\b',                     'nike'),
    (r'\bacg\b',                        'nike acg'),
    (r'\bnike ?pro\b',                  'nike pro'),

    # ══════════════════════════════════════════════════════════════
    #  JORDAN / AIR JORDAN
    # ══════════════════════════════════════════════════════════════
    (r'\baj[ -]?(\d+)\b',               r'air jordan \1'),
    (r'\bj[ -]?1\b',                    'jordan 1'),
    (r'\bj[ -]?2\b',                    'jordan 2'),
    (r'\bj[ -]?3\b',                    'jordan 3'),
    (r'\bj[ -]?4\b',                    'jordan 4'),
    (r'\bj[ -]?5\b',                    'jordan 5'),
    (r'\bj[ -]?6\b',                    'jordan 6'),
    (r'\bj[ -]?7\b',                    'jordan 7'),
    (r'\bj[ -]?8\b',                    'jordan 8'),
    (r'\bj[ -]?9\b',                    'jordan 9'),
    (r'\bj[ -]?10\b',                   'jordan 10'),
    (r'\bj[ -]?11\b',                   'jordan 11'),
    (r'\bj[ -]?12\b',                   'jordan 12'),
    (r'\bj[ -]?13\b',                   'jordan 13'),
    (r'\bj[ -]?14\b',                   'jordan 14'),
    (r'\bjordan ?low\b',                'jordan 1 low'),
    (r'\bjordan ?mid\b',                'jordan 1 mid'),
    (r'\bjordan ?high\b',               'jordan 1 high'),
    (r'\bts[ -]?j(?:ordan)?[ -]?1\b',   'travis scott jordan 1'),
    (r'\bts[ -]?j(?:ordan)?[ -]?4\b',   'travis scott jordan 4'),
    (r'\bts[ -]?j(?:ordan)?[ -]?6\b',   'travis scott jordan 6'),
    (r'\bbred\b',                       'jordan bred'),
    (r'\broyals?\b',                    'jordan royal'),
    (r'\bchicago\b',                    'jordan 1 chicago'),
    (r'\bmocha(s?)\b',                  'jordan 1 mocha'),
    (r'\buniversity ?blue\b',           'jordan 1 university blue'),
    (r'\bshadow(s?)\b',                'jordan 1 shadow'),
    (r'\bobsidian\b',                   'jordan 1 obsidian'),
    (r'\bblack ?cat\b',                 'jordan 4 black cat'),
    (r'\bmilitary ?black\b',            'jordan 4 military black'),
    (r'\bmilitary ?blue\b',             'jordan 4 military blue'),
    (r'\bfire ?red\b',                  'jordan 4 fire red'),
    (r'\bconcord\b',                    'jordan 11 concord'),
    (r'\bspace ?jam\b',                 'jordan 11 space jam'),
    (r'\bcool ?grey\b',                 'jordan 11 cool grey'),
    (r'\bcements?\b',                   'jordan 4 cement'),

    # ══════════════════════════════════════════════════════════════
    #  ADIDAS
    # ══════════════════════════════════════════════════════════════
    (r'\bgazelle(s?)\b',                'adidas gazelle'),
    (r'\bsamba(s?)\b',                  'adidas samba'),
    (r'\bsamba ?og\b',                  'adidas samba og'),
    (r'\bspezial\b',                    'adidas spezial'),
    (r'\bhandball ?spezial\b',          'adidas handball spezial'),
    (r'\bcampus(es)?\b',               'adidas campus'),
    (r'\bcampus ?00s?\b',               'adidas campus 00s'),
    (r'\bsuperstar(s?)\b',             'adidas superstar'),
    (r'\bstan ?smith\b',                'adidas stan smith'),
    (r'\bforum(s?)\b',                  'adidas forum'),
    (r'\bforum ?low\b',                 'adidas forum low'),
    (r'\bnmds?\b',                      'adidas nmd'),
    (r'\bnmd ?r1\b',                    'adidas nmd r1'),
    (r'\bnmd ?v2\b',                    'adidas nmd v2'),
    (r'\bultra ?boost\b',              'adidas ultraboost'),
    (r'\bub\b',                         'adidas ultraboost'),
    (r'\b4d\b',                         'adidas 4d'),
    (r'\bozweego\b',                    'adidas ozweego'),
    (r'\bresponse ?cl\b',               'adidas response cl'),
    (r'\bad[i1] ?2000\b',               'adidas adi2000'),
    (r'\briva ?le\b',                   'adidas rivalry'),
    (r'\brivalry\b',                    'adidas rivalry'),
    (r'\bsolar ?glide\b',               'adidas solar glide'),
    (r'\bsl[ -]?72\b',                  'adidas sl 72'),

    # ══════════════════════════════════════════════════════════════
    #  YEEZY (Adidas / Kanye)
    # ══════════════════════════════════════════════════════════════
    (r'\byzy\b',                        'yeezy'),
    (r'\byeezy ?350\b',                 'yeezy 350'),
    (r'\b350 ?v2\b',                    'yeezy 350 v2'),
    (r'\byeezy ?500\b',                 'yeezy 500'),
    (r'\byeezy ?700\b',                 'yeezy 700'),
    (r'\b700 ?v2\b',                    'yeezy 700 v2'),
    (r'\b700 ?v3\b',                    'yeezy 700 v3'),
    (r'\byeezy ?slide(s?)\b',           'yeezy slide'),
    (r'\bslide(s?)\b',                  'yeezy slide'),
    (r'\byeezy ?foam\b',               'yeezy foam runner'),
    (r'\bfoam ?rnnr\b',                 'yeezy foam runner'),
    (r'\bfoam ?runner\b',               'yeezy foam runner'),
    (r'\byeezy ?boost\b',              'yeezy boost'),
    (r'\byeezy ?380\b',                 'yeezy 380'),
    (r'\byeezy ?450\b',                 'yeezy 450'),
    (r'\byeezy ?knit\b',                'yeezy knit runner'),
    (r'\bwaverunner\b',                 'yeezy 700 wave runner'),
    (r'\bwave ?runner\b',               'yeezy 700 wave runner'),
    (r'\bzebra(s?)\b',                  'yeezy 350 zebra'),
    (r'\bbutter\b',                     'yeezy 350 butter'),
    (r'\bbeluga\b',                     'yeezy 350 beluga'),
    (r'\bstatic(s?)\b',                 'yeezy 350 static'),
    (r'\bbred ?350\b',                  'yeezy 350 bred'),

    # ══════════════════════════════════════════════════════════════
    #  NEW BALANCE
    # ══════════════════════════════════════════════════════════════
    (r'\bnb\b',                         'new balance'),
    (r'\b327s?\b',                      'new balance 327'),
    (r'\b530s?\b',                      'new balance 530'),
    (r'\b550s?\b',                      'new balance 550'),
    (r'\b574s?\b',                      'new balance 574'),
    (r'\b990s?\b',                      'new balance 990'),
    (r'\b990 ?v3\b',                    'new balance 990v3'),
    (r'\b990 ?v4\b',                    'new balance 990v4'),
    (r'\b990 ?v5\b',                    'new balance 990v5'),
    (r'\b990 ?v6\b',                    'new balance 990v6'),
    (r'\b992s?\b',                      'new balance 992'),
    (r'\b993s?\b',                      'new balance 993'),
    (r'\b1906\b',                       'new balance 1906'),
    (r'\b1906r\b',                      'new balance 1906r'),
    (r'\b2002r\b',                      'new balance 2002r'),
    (r'\b9060\b',                       'new balance 9060'),
    (r'\b1080\b',                       'new balance 1080'),
    (r'\bjoe ?freshgoods\b',            'new balance joe freshgoods'),
    (r'\baime ?leon\b',                 'new balance aime leon dore'),
    (r'\bald\b',                        'new balance aime leon dore'),

    # ══════════════════════════════════════════════════════════════
    #  BALENCIAGA
    # ══════════════════════════════════════════════════════════════
    (r'\bbalenciaga\b',                 'balenciaga'),
    (r'\bbalenci\b',                    'balenciaga'),
    (r'\bbalen\b',                      'balenciaga'),
    (r'\btriple[ -]?s\b',              'balenciaga triple s'),
    (r'\btrack(s?)\b',                  'balenciaga track'),
    (r'\btrack ?runner\b',              'balenciaga track'),
    (r'\bspeed ?trainer\b',            'balenciaga speed trainer'),
    (r'\bspeed ?runner\b',              'balenciaga speed trainer'),
    (r'\bdefender\b',                   'balenciaga defender'),
    (r'\b3xl\b',                        'balenciaga 3xl'),
    (r'\bparis ?sneaker\b',             'balenciaga paris'),
    (r'\brunner(s?)\b',                 'balenciaga runner'),

    # ══════════════════════════════════════════════════════════════
    #  LUXURY — Louis Vuitton, Gucci, Dior, Prada, etc.
    # ══════════════════════════════════════════════════════════════
    (r'\blv\b',                         'louis vuitton'),
    (r'\blouis\b',                      'louis vuitton'),
    (r'\bvuitton\b',                    'louis vuitton'),
    (r'\blv ?trainer\b',                'louis vuitton trainer'),
    (r'\blv ?archlight\b',              'louis vuitton archlight'),
    (r'\blv ?skate\b',                  'louis vuitton skate'),
    (r'\bneverfull\b',                  'louis vuitton neverfull'),
    (r'\bspeedy\b',                     'louis vuitton speedy'),
    (r'\bkeepall\b',                    'louis vuitton keepall'),
    (r'\bmonogram\b',                   'louis vuitton monogram'),
    (r'\bdamier\b',                     'louis vuitton damier'),

    (r'\bgucci\b',                      'gucci'),
    (r'\bace(s?)\b',                    'gucci ace'),
    (r'\brhyton\b',                     'gucci rhyton'),
    (r'\bscreener\b',                   'gucci screener'),
    (r'\bgg\b',                         'gucci'),

    (r'\bdior\b',                       'dior'),
    (r'\bb23\b',                        'dior b23'),
    (r'\bb22\b',                        'dior b22'),
    (r'\bdior ?jordan\b',               'dior jordan 1'),
    (r'\bsaddle\b',                     'dior saddle bag'),
    (r'\bbody ?oblique\b',              'dior oblique'),

    (r'\bprada\b',                      'prada'),
    (r'\bcloudbust\b',                  'prada cloudbust'),
    (r'\bprada ?america\b',             'prada america cup'),
    (r'\bamerica ?cup\b',               'prada america cup'),
    (r'\bre ?nylon\b',                  'prada re-nylon'),

    (r'\bversace\b',                    'versace'),
    (r'\bchain ?reaction\b',            'versace chain reaction'),

    (r'\bhermes\b',                     'hermes'),
    (r'\bhermès\b',                     'hermes'),
    (r'\boran\b',                       'hermes oran'),
    (r'\bbirkin\b',                     'hermes birkin'),
    (r'\bkelly\b',                      'hermes kelly'),

    (r'\bbottega\b',                    'bottega veneta'),
    (r'\bbv\b',                         'bottega veneta'),
    (r'\bpuddle ?boot\b',               'bottega veneta puddle boot'),
    (r'\blug ?boot\b',                  'bottega veneta lug boot'),

    (r'\bmiu ?miu\b',                   'miu miu'),

    (r'\bmaison ?margiela\b',           'maison margiela'),
    (r'\bmargiela\b',                   'maison margiela'),
    (r'\bmm6\b',                        'maison margiela'),
    (r'\bmm ?replica\b',                'maison margiela replica'),
    (r'\btabi\b',                       'maison margiela tabi'),

    (r'\bmcqueen\b',                    'alexander mcqueen'),
    (r'\bamq\b',                        'alexander mcqueen'),
    (r'\boversized ?sneaker\b',          'alexander mcqueen oversized'),

    (r'\bvalentino\b',                  'valentino'),
    (r'\brockstud\b',                   'valentino rockstud'),

    (r'\bburberry\b',                   'burberry'),
    (r'\bceline\b',                     'celine'),
    (r'\bfendi\b',                      'fendi'),
    (r'\bysl\b',                        'yves saint laurent'),
    (r'\bsaint ?laurent\b',             'yves saint laurent'),

    # ══════════════════════════════════════════════════════════════
    #  STREETWEAR — Off-White, Travis Scott, FOG, Supreme, etc.
    # ══════════════════════════════════════════════════════════════
    (r'\bts\b',                         'travis scott'),
    (r'\btravis\b',                     'travis scott'),
    (r'\bcactus ?jack\b',               'travis scott cactus jack'),
    (r'\boff ?white\b',                 'off white'),
    (r'\bow\b',                         'off white'),
    (r'\bvirgil\b',                     'off white'),

    (r'\bfog\b',                        'fear of god'),
    (r'\bessentials?\b',                'fear of god essentials'),
    (r'\bfear ?of ?god\b',              'fear of god'),

    (r'\bsupremo?\b',                   'supreme'),
    (r'\bsupreme\b',                    'supreme'),
    (r'\bbogo\b',                       'supreme box logo'),
    (r'\bbox ?logo\b',                  'supreme box logo'),

    (r'\bstussy\b',                     'stussy'),
    (r'\btrapstar\b',                   'trapstar'),
    (r'\bcorteiz\b',                    'corteiz'),
    (r'\bcrtz\b',                       'corteiz'),

    (r'\bpalace\b',                     'palace'),
    (r'\bbape\b',                       'bape'),
    (r'\bbathing ?ape\b',               'bape'),
    (r'\bape ?bapesta\b',               'bape bapesta'),
    (r'\bbapesta(s?)\b',                'bape bapesta'),

    (r'\bkith\b',                       'kith'),
    (r'\bunion ?la\b',                  'union los angeles'),
    (r'\bvlone\b',                      'vlone'),
    (r'\brhude\b',                      'rhude'),
    (r'\bgallery ?dept\b',              'gallery dept'),
    (r'\bgd\b',                         'gallery dept'),
    (r'\brepresent\b',                  'represent'),
    (r'\bamiri\b',                      'amiri'),
    (r'\bchromes? ?hearts?\b',          'chrome hearts'),
    (r'\bch\b',                         'chrome hearts'),
    (r'\bhuman ?made\b',                'human made'),
    (r'\bnigo\b',                       'human made'),

    # ══════════════════════════════════════════════════════════════
    #  ASICS / ONITSUKA / PUMA / REEBOK / CONVERSE / VANS
    # ══════════════════════════════════════════════════════════════
    (r'\bono?tsuka\b',                  'onitsuka tiger'),
    (r'\bmexico ?66\b',                 'onitsuka tiger mexico 66'),
    (r'\bgel[ -]?kayano\b',            'asics gel kayano'),
    (r'\bgel[ -]?1130\b',               'asics gel 1130'),
    (r'\bgel[ -]?nyc\b',                'asics gel nyc'),
    (r'\bgel[ -]?lyte\b',               'asics gel lyte'),
    (r'\bgt[ -]?2160\b',                'asics gt 2160'),
    (r'\basics\b',                      'asics'),

    (r'\bpuma\b',                       'puma'),
    (r'\bsuede(s?)\b',                  'puma suede'),
    (r'\brs[ -]?x\b',                   'puma rs-x'),

    (r'\breebok\b',                     'reebok'),
    (r'\bclub ?c\b',                    'reebok club c'),
    (r'\bclassic ?leather\b',           'reebok classic leather'),
    (r'\binstapump\b',                  'reebok instapump fury'),

    (r'\bconverse\b',                   'converse'),
    (r'\bchuck(s?)\b',                  'converse chuck taylor'),
    (r'\bchuck ?70\b',                  'converse chuck 70'),
    (r'\ball ?star\b',                   'converse all star'),
    (r'\bcdg ?converse\b',              'comme des garcons converse'),

    (r'\bvans\b',                       'vans'),
    (r'\bold ?skool\b',                 'vans old skool'),
    (r'\bsk8[ -]?hi\b',                'vans sk8 hi'),
    (r'\bera\b',                        'vans era'),

    # ══════════════════════════════════════════════════════════════
    #  DESIGNER BRANDS — Misc
    # ══════════════════════════════════════════════════════════════
    (r'\brick ?owens?\b',               'rick owens'),
    (r'\bricks?\b',                     'rick owens'),
    (r'\bramones?\b',                   'rick owens ramones'),
    (r'\bgeobasket\b',                  'rick owens geobasket'),
    (r'\bjumbo ?lace\b',                'rick owens jumbo lace'),

    (r'\bsalomon\b',                    'salomon'),
    (r'\bxt[ -]?6\b',                   'salomon xt-6'),
    (r'\bxt[ -]?4\b',                   'salomon xt-4'),
    (r'\bacs ?pro\b',                   'salomon acs pro'),

    (r'\bhoka\b',                       'hoka'),
    (r'\bclifton\b',                    'hoka clifton'),
    (r'\bbondi\b',                      'hoka bondi'),
    (r'\bmafate\b',                     'hoka mafate'),

    (r'\bon ?cloud\b',                  'on cloud'),
    (r'\bcloudmonster\b',               'on cloudmonster'),

    (r'\bcrocs?\b',                     'crocs'),
    (r'\bbirkens?\b',                   'birkenstock'),
    (r'\bbirkenstock\b',                'birkenstock'),
    (r'\bboston\b',                     'birkenstock boston'),

    (r'\bugg(s?)\b',                    'ugg'),
    (r'\btasman\b',                     'ugg tasman'),
    (r'\bultra ?mini\b',                'ugg ultra mini'),

    (r'\btimbs?\b',                     'timberland'),
    (r'\btimberland\b',                 'timberland'),

    (r'\bdr ?martens?\b',               'dr martens'),
    (r'\bdocs?\b',                      'dr martens'),

    # ══════════════════════════════════════════════════════════════
    #  ACCESORIOS & BOLSOS
    # ══════════════════════════════════════════════════════════════
    (r'\bbackpack\b',                   'backpack'),
    (r'\bmochila\b',                    'backpack'),
    (r'\bcross ?body\b',                'crossbody bag'),
    (r'\bbelt ?bag\b',                  'belt bag'),
    (r'\btote\b',                       'tote bag'),
    (r'\bfanny ?pack\b',                'fanny pack'),
    (r'\briñonera\b',                   'fanny pack'),
    (r'\bcartera\b',                    'wallet'),
    (r'\bwallet\b',                     'wallet'),
    (r'\bgorra\b',                      'cap hat'),
    (r'\bcap\b',                        'cap hat'),
    (r'\bsunglasses\b',                 'sunglasses'),
    (r'\blentes\b',                     'sunglasses'),
    (r'\bgafas\b',                      'sunglasses'),
    (r'\bbelt\b',                       'belt'),
    (r'\bcintur[oó]n\b',                'belt'),
    (r'\bwatch\b',                      'watch'),
    (r'\breloj\b',                      'watch'),
    (r'\bjewelry\b',                    'jewelry'),
    (r'\bcadena\b',                     'chain necklace'),
    (r'\bnecklace\b',                   'chain necklace'),
    (r'\bpendant\b',                    'pendant necklace'),
    (r'\bbracelet\b',                   'bracelet'),
    (r'\bpulsera\b',                    'bracelet'),
    (r'\bring\b',                       'ring'),
    (r'\banillo\b',                     'ring'),

    # ══════════════════════════════════════════════════════════════
    #  ROPA / CLOTHING
    # ══════════════════════════════════════════════════════════════
    (r'\bhoodie(s?)\b',                 'hoodie'),
    (r'\bsudadera\b',                   'hoodie'),
    (r'\bcrew ?neck\b',                 'crewneck sweatshirt'),
    (r'\bsweatpants?\b',                'sweatpants'),
    (r'\bjogger(s?)\b',                 'joggers'),
    (r'\bcargo(s?)\b',                  'cargo pants'),
    (r'\bpuffer\b',                     'puffer jacket'),
    (r'\bdown ?jacket\b',               'puffer jacket'),
    (r'\bplumon\b',                     'puffer jacket'),
    (r'\bnorth ?face\b',                'the north face'),
    (r'\btnf\b',                        'the north face'),
    (r'\bnuptse\b',                     'the north face nuptse'),
    (r'\barcteryx\b',                   'arcteryx'),
    (r'\barc ?teryx\b',                 'arcteryx'),
    (r'\bmoncler\b',                    'moncler'),
    (r'\bcanada ?goose\b',              'canada goose'),
    (r'\bstone ?island\b',              'stone island'),
    (r'\bsi\b',                         'stone island'),
    (r'\bcp ?company\b',                'cp company'),
    (r'\bpatagonia\b',                  'patagonia'),

    (r'\bpolo\b',                       'polo ralph lauren'),
    (r'\bralph ?lauren\b',              'polo ralph lauren'),
    (r'\brl\b',                         'polo ralph lauren'),
    (r'\blacoste\b',                    'lacoste'),
    (r'\btommy\b',                      'tommy hilfiger'),
    (r'\bcalvin ?klein\b',              'calvin klein'),
    (r'\bck\b',                         'calvin klein'),
]

# Stopwords que no ayudan en búsqueda
STOPWORDS = frozenset({
    'de', 'la', 'el', 'las', 'los', 'un', 'una', 'unos', 'unas', 'me',
    'te', 'se', 'en', 'con', 'por', 'para', 'que', 'del', 'al', 'y',
    'o', 'a', 'es', 'son', 'hay', 'tiene', 'ver', 'quiero', 'dame',
    'muestra', 'enseña', 'busca', 'buscame', 'encuentra', 'the', 'of',
    'estos', 'estas', 'como', 'donde', 'puedo', 'puedes', 'tienes',
    'mira', 'necesito', 'algo', 'unas', 'unos', 'tipo',
})


def normalize(q: str) -> str:
    """Normaliza: minúsculas, aplica aliases, limpia caracteres raros."""
    q = q.lower().strip()
    q = re.sub(r'[^\w\s]', ' ', q)  # Remove punctuation
    for pat, rep in ALIASES:
        q = re.sub(pat, rep, q)
    return ' '.join(q.split())  # Normalize whitespace


def generate_variants(query: str) -> list[str]:
    """Genera variantes inteligentes del query para búsqueda más amplia."""
    norm = normalize(query)
    words = [w for w in norm.split() if w not in STOPWORDS and len(w) > 1]

    variants: set[str] = set()
    full = ' '.join(words)
    if full:
        variants.add(full)

    if len(words) > 2:
        # Brand + model (first + last word)
        variants.add(f"{words[0]} {words[-1]}")
        # Without first word (in case it's a generic brand)
        variants.add(' '.join(words[1:]))
        # Without last word
        variants.add(' '.join(words[:-1]))

    if len(words) == 2:
        for w in words:
            if len(w) > 3:
                variants.add(w)

    # If we only have 1 word, keep it as is
    return list(variants) if variants else [query.lower().strip()]


def clean_title(t: str) -> str:
    """Remove HTML tags from title."""
    return re.sub(r'<[^>]+>', '', t).strip()


def fuzzy_match(word: str, title: str) -> float:
    """Fuzzy matching for typo-tolerant search."""
    if word in title:
        return 1.0
    title_words = title.split()
    best = 0.0
    for tw in title_words:
        ratio = SequenceMatcher(None, word, tw).ratio()
        if ratio > best:
            best = ratio
    return best


def score_product(p: dict, words: list[str]) -> float:
    """Advanced scoring system combining relevance and quality signals."""
    title = clean_title(p.get('title', '')).lower()

    if not words:
        return 0.0

    # ── 1. Relevance Score ──────────────────────────────────────────
    exact_hits = sum(1 for w in words if w in title)
    fuzzy_scores = [fuzzy_match(w, title) for w in words if w not in title]
    fuzzy_hits = sum(s for s in fuzzy_scores if s >= 0.7)  # Only count decent fuzzy matches

    total_hits = exact_hits + (fuzzy_hits * 0.5)
    relevance = total_hits / len(words)  # 0.0 to 1.0

    # Skip products with very low relevance
    if relevance < 0.3:
        return 0.0

    # ── 2. Quality Score (0-100 range) ──────────────────────────────
    star = p.get('star', 0) or 0
    qc_count = p.get('qcImageCount', 0) or 0

    # Star rating (max 30 points)
    if star >= 4.5:
        star_score = 30
    elif star >= 4.0:
        star_score = 22
    elif star >= 3.5:
        star_score = 15
    elif star > 0:
        star_score = star * 3
    else:
        star_score = 0

    # QC photos (max 25 points) — more photos = more trustworthy
    qc_score = min(qc_count / 10, 25)

    # Discount bonus (max 10 points)
    discount_price = p.get('discountPrice')
    original_price = p.get('price', 0)
    discount_score = 0
    if discount_price and original_price and discount_price < original_price:
        savings_pct = ((original_price - discount_price) / original_price) * 100
        discount_score = min(savings_pct / 5, 10)

    # Penalty for no reviews (risky purchase)
    no_review_penalty = -15 if star == 0 and qc_count == 0 else 0

    # New product bonus
    new_bonus = 5 if p.get('newFlag') == 1 else 0

    quality = star_score + qc_score + discount_score + no_review_penalty + new_bonus

    # ── 3. Final Score ──────────────────────────────────────────────
    # Relevance is the dominant factor (70%), quality is secondary (30%)
    return (relevance * 70) + (quality * 0.3)


def build_url(query: str, page: int = 1, page_size: int = 40, sort: str = '') -> str:
    """Build the Hipopick/Plug4me API search URL."""
    params = {
        'pageNo': page,
        'pageSize': page_size,
        'productName': query,
        'sort': sort,
        'channel': 'local',
    }
    return 'https://gateway.plug4.me/goods-service/spu/page?' + urllib.parse.urlencode(params)


def parse_product(p: dict) -> dict:
    """Extract all useful product info including QC photos."""
    price = p.get('discountPrice') or p.get('price', 0)
    orig = p.get('price', 0) if p.get('discountPrice') else None
    wid = p.get('channelItemNo', '')

    # Extract ALL available QC photos
    qc_images = []
    qc_groups = p.get('qcImageVoGroupMap')
    if isinstance(qc_groups, dict):
        for _group_name, group in qc_groups.items():
            if isinstance(group, list):
                for img in group[:5]:  # Max 5 per group
                    url = ''
                    if isinstance(img, dict):
                        url = img.get('imageUrl', '') or img.get('webpUrl', '')
                    if url:
                        full_url = url if url.startswith('http') else QC_BASE + url
                        qc_images.append(full_url)

    # Calculate savings percentage
    savings = None
    if orig and price and orig > price:
        savings = round(((orig - price) / orig) * 100)

    return {
        'title':     clean_title(p.get('title', '')),
        'price':     price,
        'orig':      orig,
        'savings':   savings,
        'image':     p.get('mainImage', ''),
        'star':      p.get('star', 0) or 0,
        'qc_count':  p.get('qcImageCount', 0) or 0,
        'qc_images': qc_images,
        'qc_img':    qc_images[0] if qc_images else None,
        'shop':      p.get('shopName', '') or 'Tienda',
        'is_new':    p.get('newFlag') == 1,
        'wid':       wid,
        'source':    p.get('sourceUrl', ''),
        'url_plug':  f'https://plug4.me/item/{wid}?channel=weidian',
        'url_hipo':  f'https://hipobuy.com/product/2/{wid}',
        'url_hipos': f'https://hipobuys.com/index/detail?id={wid}&channel=weidian',
    }


async def do_search(query: str, sort: str = '', top_n: int = 5) -> list[dict]:
    """Intelligent search with variants, deduplication, and caching."""

    # Check cache first
    cached = search_cache.get(query, sort, top_n)
    if cached is not None:
        return cached

    variants = generate_variants(query)
    all_products: dict[str, tuple[float, dict]] = {}  # wid -> (score, raw_product)

    words = [w for w in normalize(query).split() if w not in STOPWORDS and len(w) > 1]

    for variant in variants[:3]:  # Max 3 variants
        try:
            url = build_url(variant, sort=sort)
            resp = await api_request_with_retry("GET", url, retries=1)
            data = resp.json()

            if not data.get('success'):
                logger.warning(f"API returned success=false for variant '{variant}'")
                continue

            records = data.get('result', {}).get('records', [])
            for p in records:
                wid = p.get('channelItemNo', '')
                if not wid:
                    continue
                s = score_product(p, words)
                if s <= 0:
                    continue
                # Keep the best score if duplicates
                if wid not in all_products or s > all_products[wid][0]:
                    all_products[wid] = (s, p)

        except Exception as e:
            logger.warning(f"Error searching variant '{variant}': {e}")
            continue

    if not all_products:
        return []

    # Sort by score descending, take top_n
    sorted_products = sorted(all_products.values(), key=lambda x: -x[0])
    results = [parse_product(p) for _, p in sorted_products[:top_n]]

    # Cache the results
    search_cache.put(query, sort, top_n, results)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCT DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def fmt_product(p: dict, idx: int) -> str:
    """Format product info for Telegram with safe Markdown."""
    title = escape_md(p['title'][:60] + ('…' if len(p['title']) > 60 else ''))

    # Price line
    price_str = f"💰 ${p['price']:.2f}"
    if p.get('savings') and p.get('orig'):
        price_str += f"  ~${p['orig']:.2f}~  🔥 -{p['savings']}%"
    elif p.get('orig') and p['orig'] > p['price']:
        price_str += f"  ~${p['orig']:.2f}~"

    # Star rating
    star = p.get('star', 0)
    if star:
        filled = '⭐' * min(int(star), 5)
        stars = f"{filled} {star:.1f}"
    else:
        stars = "🔘 Sin reseñas"

    # QC count
    qc_count = p.get('qc_count', 0)
    qc = f"📸 {qc_count} fotos QC" if qc_count else "📸 Sin QC aún"

    # Badges
    badges = []
    if p.get('is_new'):
        badges.append("🆕")
    if p.get('savings') and p['savings'] >= 20:
        badges.append("🔥 OFERTA")
    if star >= 4.5:
        badges.append("👑 TOP")
    if qc_count >= 100:
        badges.append("✅ VERIFICADO")
    badge_str = ' '.join(badges)

    shop_name = escape_md(p.get('shop', 'Tienda'))

    lines = [f"{idx}. {title}"]
    if badge_str:
        lines.append(badge_str)
    lines.append(price_str)
    lines.append(f"{stars} | {qc}")
    lines.append(f"🏪 {shop_name}")

    return '\n'.join(lines)


def product_kb(p: dict) -> InlineKeyboardMarkup:
    """Inline keyboard with buy links and QC photos."""
    row1 = []
    row2 = [
        InlineKeyboardButton("🔗 Plug4.me", url=p['url_plug']),
        InlineKeyboardButton("🛒 Hipobuy", url=p['url_hipo']),
        InlineKeyboardButton("📋 Hipobuys", url=p['url_hipos']),
    ]

    # QC photo buttons
    if p.get('qc_img'):
        row1.append(InlineKeyboardButton("📸 Ver QC", url=p['qc_img']))
    if len(p.get('qc_images', [])) > 1:
        row1.append(
            InlineKeyboardButton(f"🖼 +{len(p['qc_images']) - 1} fotos", url=p['qc_images'][1])
        )

    keyboard = []
    if row1:
        keyboard.append(row1)
    keyboard.append(row2)

    return InlineKeyboardMarkup(keyboard)


async def send_products(update: Update, products: list[dict], header: str):
    """Send product results to the user with proper error handling."""
    try:
        await update.message.reply_text(header)
    except Exception:
        await update.message.reply_text(header.replace("*", "").replace("_", ""))

    for i, p in enumerate(products, 1):
        caption = fmt_product(p, i)
        try:
            # Try sending with photo
            await update.message.reply_photo(
                photo=p['image'],
                caption=caption,
                reply_markup=product_kb(p),
            )
        except Exception as photo_err:
            logger.warning(f"Failed to send product photo: {photo_err}")
            try:
                # Fallback: send as text only
                await update.message.reply_text(
                    caption,
                    reply_markup=product_kb(p),
                )
            except Exception as text_err:
                logger.error(f"Failed to send product text: {text_err}")
                # Last resort: plain text, no formatting
                await update.message.reply_text(
                    caption.replace("*", "").replace("_", ""),
                    reply_markup=product_kb(p),
                )


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ AI
# ══════════════════════════════════════════════════════════════════════════════

async def groq_chat(uid: int, user_msg) -> str:
    """Chat with Groq LLM. Handles both text and vision messages."""
    memory.add(uid, "user", user_msg)

    model = VISION_MODEL if isinstance(user_msg, list) else TEXT_MODEL
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + memory.get(uid)

    try:
        resp = await api_request_with_retry(
            "POST",
            GROQ_CHAT_URL,
            headers=GROQ_HEADERS,
            json={"model": model, "messages": messages, "max_tokens": 1024},
        )
        data = resp.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not reply:
            reply = "No pude generar una respuesta. Intenta de nuevo."
    except Exception as e:
        logger.error(f"Groq chat error for user {uid}: {e}")
        reply = "😔 Tengo problemas para responder ahora. Intenta de nuevo en unos segundos."

    memory.add(uid, "assistant", reply)
    return reply


async def groq_classify(text: str) -> dict:
    """Classify user intent: search or chat."""
    messages = [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        resp = await api_request_with_retry(
            "POST",
            GROQ_CHAT_URL,
            headers=GROQ_HEADERS,
            json={
                "model": TEXT_MODEL,
                "messages": messages,
                "max_tokens": 80,
                "temperature": 0,
            },
            retries=1,
        )
        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse classifier JSON: {text[:50]}")
    except Exception as e:
        logger.error(f"Classifier error: {e}")

    return {"intent": "chat"}


async def groq_transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio using Whisper via Groq."""
    try:
        client = await get_http_client()
        resp = await client.post(
            GROQ_WHISPER_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.ogg", audio_bytes, "audio/ogg")},
            data={"model": AUDIO_MODEL, "response_format": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


async def groq_describe(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Describe an image using Groq Vision model."""
    b64 = base64.b64encode(image_bytes).decode()
    content = [
        {"type": "text", "text": (
            "Describe esta imagen. Si contiene un producto (zapatilla, ropa, bolso, accesorio), "
            "identifica la MARCA y MODELO exacto (ejemplo: 'Nike Air Jordan 4 Military Blue', "
            "'Adidas Yeezy 350 Zebra', 'Louis Vuitton Neverfull'). "
            "Si hay texto visible, transcríbelo."
        )},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
    try:
        resp = await api_request_with_retry(
            "POST",
            GROQ_CHAT_URL,
            headers=GROQ_HEADERS,
            json={
                "model": VISION_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1024,
            },
        )
        return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"Vision describe error: {e}")
        return "No pude analizar la imagen."


async def groq_identify_product(description: str) -> str:
    """Extract a searchable product name from an image description."""
    messages = [
        {"role": "system", "content": (
            "Eres un experto en identificar productos de moda, zapatillas y ropa. "
            "A partir de una descripción de imagen, extrae el nombre del producto para búsqueda. "
            "Devuelve SOLO el nombre del producto en inglés, corto y buscable. "
            "Ejemplos: 'jordan 4 military blue', 'nike dunk panda', 'yeezy 350 zebra', "
            "'louis vuitton neverfull', 'nike tech fleece grey'. "
            "Si no puedes identificar el producto, devuelve 'unknown'. "
            "SOLO el nombre, nada más."
        )},
        {"role": "user", "content": f"Identifica el producto en esta descripción: {description}"},
    ]
    try:
        resp = await api_request_with_retry(
            "POST",
            GROQ_CHAT_URL,
            headers=GROQ_HEADERS,
            json={
                "model": TEXT_MODEL,
                "messages": messages,
                "max_tokens": 50,
                "temperature": 0,
            },
            retries=1,
        )
        result = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        # Clean up: remove quotes, periods
        result = result.strip('"\'.').strip()
        if result.lower() == "unknown" or len(result) < 3:
            return ""
        return result
    except Exception as e:
        logger.error(f"Product identification error: {e}")
        return ""


async def download_file(bot, file_id: str) -> bytes:
    """Download a file from Telegram servers."""
    try:
        f = await bot.get_file(file_id)
        client = await get_http_client()
        resp = await client.get(f.file_path, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.error(f"File download error: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle /start and /help commands."""
    name = escape_md(update.effective_user.first_name or "amigo")
    await update.message.reply_text(
        f"👋 Hola {name}! Soy tu asistente de reps 🤖👟\n\n"
        f"¿Qué puedo hacer?\n"
        f"🔍 Buscar productos — solo pídelo natural\n"
        f"💬 Charlar de lo que quieras\n"
        f"🎤 Entender tus audios\n"
        f"📷 Analizar fotos\n\n"
        f"Comandos:\n"
        f"/buscar [producto] — Buscar productos\n"
        f"/top [producto] — El mejor resultado\n"
        f"/precio [producto] — Los más baratos\n"
        f"/qc [producto] — Ver fotos QC\n"
        f"/reset — Limpiar conversación\n\n"
        f"Ejemplos:\n"
        f"• 'quiero ver unas jordan 4'\n"
        f"• 'búscame yeezy slides'\n"
        f"• 'qué tal están las dunk panda'\n\n"
        f"¡Escríbeme lo que necesites! 😎",
    )


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history for the user."""
    uid = update.effective_user.id
    memory.clear(uid)
    await update.message.reply_text("🗑 Conversación limpiada. ¡Empezamos de nuevo! 😎")


async def cmd_buscar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Search for products."""
    query = ' '.join(ctx.args) if ctx.args else ''
    if not query:
        await update.message.reply_text("✍️ Dime qué quieres buscar\nEj: nike dunk panda")
        return

    uid = update.effective_user.id
    if not rate_limiter.check(uid):
        remaining = rate_limiter.remaining(uid)
        await update.message.reply_text(
            f"⏳ Espera {remaining:.0f}s antes de buscar de nuevo."
        )
        return

    await update.message.chat.send_action("typing")
    try:
        products = await do_search(query, top_n=5)
        if not products:
            await update.message.reply_text(
                f"😕 No encontré resultados para '{escape_md(query)}'.\n"
                f"💡 Prueba con otro nombre o en inglés."
            )
            return
        await send_products(
            update, products,
            f"✅ Top {len(products)} resultados para \"{escape_md(query)}\":\n"
            f"🏷 Ordenados por relevancia y calidad"
        )
    except Exception as e:
        logger.error(f"Search error for '{query}': {e}")
        await update.message.reply_text("❌ Error al buscar. Intenta de nuevo en unos segundos.")


async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Get the single best product."""
    query = ' '.join(ctx.args) if ctx.args else ''
    if not query:
        await update.message.reply_text("✍️ Ej: yeezy 350")
        return

    uid = update.effective_user.id
    if not rate_limiter.check(uid):
        remaining = rate_limiter.remaining(uid)
        await update.message.reply_text(f"⏳ Espera {remaining:.0f}s.")
        return

    await update.message.chat.send_action("typing")
    try:
        products = await do_search(query, top_n=1)
        if not products:
            await update.message.reply_text("😕 No encontré resultados.")
            return
        await send_products(update, products, f"🏆 Mejor resultado para \"{escape_md(query)}\":")
    except Exception as e:
        logger.error(f"Top search error: {e}")
        await update.message.reply_text("❌ Error al buscar.")


async def cmd_precio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Search for cheapest products."""
    query = ' '.join(ctx.args) if ctx.args else ''
    if not query:
        await update.message.reply_text("✍️ Ej: jordan 1")
        return

    uid = update.effective_user.id
    if not rate_limiter.check(uid):
        remaining = rate_limiter.remaining(uid)
        await update.message.reply_text(f"⏳ Espera {remaining:.0f}s.")
        return

    await update.message.chat.send_action("typing")
    try:
        products = await do_search(query, sort='price_asc', top_n=5)
        if not products:
            await update.message.reply_text("😕 No encontré resultados.")
            return
        await send_products(update, products, f"💸 Más baratos para \"{escape_md(query)}\":")
    except Exception as e:
        logger.error(f"Price search error: {e}")
        await update.message.reply_text("❌ Error al buscar.")


async def cmd_qc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show QC photos for a product."""
    query = ' '.join(ctx.args) if ctx.args else ''
    if not query:
        await update.message.reply_text("✍️ Ej: /qc nike dunk")
        return

    uid = update.effective_user.id
    if not rate_limiter.check(uid):
        remaining = rate_limiter.remaining(uid)
        await update.message.reply_text(f"⏳ Espera {remaining:.0f}s.")
        return

    await update.message.chat.send_action("typing")
    try:
        products = await do_search(query, top_n=1)
        if not products:
            await update.message.reply_text("😕 No encontré resultados.")
            return

        p = products[0]
        qc_imgs = p.get('qc_images', [])
        title_safe = escape_md(p['title'][:50])

        if not qc_imgs:
            await update.message.reply_text(
                f"📸 {title_safe} no tiene fotos QC aún.\n"
                f"Puedes ver el producto aquí: {p['url_plug']}"
            )
            return

        num_showing = min(len(qc_imgs), 8)
        await update.message.reply_text(
            f"📸 Fotos QC de: {title_safe}\n"
            f"⭐ {p['star']:.1f} | 🏪 {escape_md(p['shop'])}\n"
            f"Mostrando {num_showing} de {p['qc_count']} fotos:"
        )
        for img_url in qc_imgs[:8]:
            try:
                await update.message.reply_photo(photo=img_url)
            except Exception:
                await update.message.reply_text(f"🔗 {img_url}")

    except Exception as e:
        logger.error(f"QC search error: {e}")
        await update.message.reply_text("❌ Error al buscar fotos QC.")


# ══════════════════════════════════════════════════════════════════════════════
#  MESSAGE HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle all text messages: classify intent and route accordingly."""
    uid = update.effective_user.id
    text = update.message.text

    if not text or not text.strip():
        return

    await update.message.chat.send_action("typing")

    try:
        # Step 1: Classify intent
        classification = await groq_classify(text)
        intent = classification.get("intent", "chat")
        query = classification.get("query", "").strip()

        if intent == "search" and len(query) > 2:
            # Rate limit check for search
            if not rate_limiter.check(uid):
                remaining = rate_limiter.remaining(uid)
                await update.message.reply_text(f"⏳ Espera {remaining:.0f}s antes de buscar.")
                return
            # Route to search
            ctx.args = query.split()
            await cmd_buscar(update, ctx)
        else:
            # Normal chat
            reply = await groq_chat(uid, text)
            reply = safe_markdown(reply)
            try:
                await update.message.reply_text(reply, parse_mode="Markdown")
            except Exception:
                # If Markdown fails, send without formatting
                await update.message.reply_text(reply.replace("*", "").replace("_", ""))

    except Exception as e:
        logger.error(f"handle_text error for user {uid}: {e}")
        await update.message.reply_text("😔 Algo salió mal, inténtalo de nuevo.")


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages: transcribe, classify intent, route."""
    uid = update.effective_user.id
    await update.message.chat.send_action("typing")

    try:
        voice = update.message.voice or update.message.audio
        if not voice:
            await update.message.reply_text("No pude procesar el audio.")
            return

        audio_bytes = await download_file(ctx.bot, voice.file_id)
        transcript = await groq_transcribe(audio_bytes)

        if not transcript:
            await update.message.reply_text("🎤 No pude entender el audio. Intenta de nuevo.")
            return

        transcript_safe = escape_md(transcript)

        # ── NEW: Classify transcription intent ──
        classification = await groq_classify(transcript)
        intent = classification.get("intent", "chat")
        query = classification.get("query", "").strip()

        if intent == "search" and len(query) > 2:
            # User asked for a search via voice!
            if not rate_limiter.check(uid):
                remaining = rate_limiter.remaining(uid)
                await update.message.reply_text(f"⏳ Espera {remaining:.0f}s.")
                return

            await update.message.reply_text(f"🎤 Escuché: {transcript_safe}\n🔍 Buscando...")
            ctx.args = query.split()
            await cmd_buscar(update, ctx)
        else:
            # Normal chat from voice
            reply = await groq_chat(uid, f"[Audio del usuario]: {transcript}")
            reply = safe_markdown(reply)
            try:
                await update.message.reply_text(
                    f"🎤 Transcripción: {transcript_safe}\n\n{reply}",
                    parse_mode="Markdown",
                )
            except Exception:
                await update.message.reply_text(
                    f"🎤 Transcripción: {transcript}\n\n{reply.replace('*', '').replace('_', '')}"
                )

    except Exception as e:
        logger.error(f"handle_voice error for user {uid}: {e}")
        await update.message.reply_text("😔 Error procesando el audio.")


# Palabras clave que indican que el usuario quiere buscar un producto de la foto
_PHOTO_SEARCH_KEYWORDS = {
    'busca', 'buscame', 'buscar', 'encuentra', 'encontrar', 'quiero',
    'estas', 'estos', 'esas', 'esos', 'asi', 'así', 'haci', 'asì',
    'donde', 'dónde', 'consigo', 'conseguir', 'comprar', 'compro',
    'tienen', 'tienes', 'hay', 'link', 'enlace', 'w2c', 'wtc',
    'find', 'search', 'cop', 'get', 'want', 'need',
}


def _caption_wants_search(caption: str) -> bool:
    """Check if a photo caption implies the user wants to search for the product."""
    if not caption:
        return False
    words = set(re.sub(r'[^\w\s]', '', caption.lower()).split())
    return bool(words & _PHOTO_SEARCH_KEYWORDS)


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle photos: identify product from image when user wants to search."""
    uid = update.effective_user.id
    await update.message.chat.send_action("typing")

    try:
        photos = update.message.photo
        if not photos:
            return

        image_bytes = await download_file(ctx.bot, photos[-1].file_id)
        desc = await groq_describe(image_bytes)
        caption = update.message.caption or ""

        # ── Check if user wants to search for the product in the photo ──
        if _caption_wants_search(caption):
            if not rate_limiter.check(uid):
                remaining = rate_limiter.remaining(uid)
                await update.message.reply_text(f"⏳ Espera {remaining:.0f}s.")
                return

            # Use AI to identify the product from the image description
            product_name = await groq_identify_product(desc)

            if product_name:
                await update.message.reply_text(
                    f"📷 Identifiqué: *{escape_md(product_name)}*\n🔍 Buscando..."
                )
                ctx.args = product_name.split()
                await cmd_buscar(update, ctx)
            else:
                # Couldn't identify — fallback to chat about the image
                await update.message.reply_text(
                    "📷 No pude identificar el producto exacto de la foto.\n"
                    "💡 Prueba escribiendo el nombre directamente, ej: _jordan 4 military blue_"
                )
            return

        # Normal image chat (no search intent)
        prompt = f"[Imagen enviada por el usuario]: {desc}"
        if caption:
            prompt += f"\n[Caption del usuario]: {caption}"
        reply = await groq_chat(uid, prompt)
        reply = safe_markdown(reply)
        try:
            await update.message.reply_text(reply, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(reply.replace("*", "").replace("_", ""))

    except Exception as e:
        logger.error(f"handle_photo error for user {uid}: {e}")
        await update.message.reply_text("😔 Error procesando la imagen.")


async def handle_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle video messages using thumbnail."""
    uid = update.effective_user.id
    await update.message.chat.send_action("typing")

    try:
        video = update.message.video
        if not video:
            return

        if video.thumbnail:
            image_bytes = await download_file(ctx.bot, video.thumbnail.file_id)
            desc = await groq_describe(image_bytes)
        else:
            desc = "un video sin miniatura disponible"

        caption = update.message.caption or ""
        prompt = f"[Frame de video]: {desc}"
        if caption:
            prompt += f"\n[Caption]: {caption}"
        reply = await groq_chat(uid, prompt)
        reply = safe_markdown(reply)
        try:
            await update.message.reply_text(reply, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(reply.replace("*", "").replace("_", ""))

    except Exception as e:
        logger.error(f"handle_video error for user {uid}: {e}")
        await update.message.reply_text("😔 Error procesando el video.")


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP & KEEP-ALIVE
# ══════════════════════════════════════════════════════════════════════════════

class HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for platform health checks."""

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is alive!")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logs


def keep_alive():
    """Start a lightweight HTTP server for platform health checks."""
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health check server running on port {port}")


async def on_shutdown(app):
    """Cleanup on bot shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        logger.info("HTTP client closed.")


async def post_init(application):
    """Runs after bot is initialized — cleans up any previous webhook/polling session."""
    try:
        await application.bot.delete_webhook(drop_pending_updates=True)
        logger.info("Webhook deleted and pending updates dropped on startup.")
    except Exception as e:
        logger.warning(f"Could not delete webhook on startup: {e}")


def get_webhook_url() -> str | None:
    """Auto-detect the external URL from common platform env vars."""
    # Check explicit setting first
    url = os.environ.get("WEBHOOK_URL", "").strip()
    if url:
        return url.rstrip("/")

    # Koyeb provides the app's public URL
    koyeb_url = os.environ.get("KOYEB_PUBLIC_DOMAIN", "").strip()
    if koyeb_url:
        return f"https://{koyeb_url}"

    # Railway provides the public domain
    railway_url = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
    if railway_url:
        return f"https://{railway_url}"

    # Render provides the external URL
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").strip()
    if render_url:
        return render_url.rstrip("/")

    return None


def main():
    """Initialize and run the bot in either POLLING or WEBHOOK mode."""

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    keep_alive()

    # Build the application with post_init to cleanup previous sessions
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    # ── Register Handlers ─────────────────────────────────────────
    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("buscar", cmd_buscar))
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("precio", cmd_precio))
    app.add_handler(CommandHandler("qc", cmd_qc))
    app.add_handler(CommandHandler("reset", cmd_reset))

    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))

    # ── Decide mode: WEBHOOK or POLLING ───────────────────────────
    deploy_mode = os.environ.get("DEPLOY_MODE", "polling").lower().strip()
    webhook_url = get_webhook_url()

    if deploy_mode == "webhook" and webhook_url:
        # ── WEBHOOK MODE ──────────────────────────────────────────
        # No polling = NO conflict errors, ever!
        port = int(os.environ.get("PORT", 8080))
        webhook_path = f"/webhook/{TELEGRAM_TOKEN}"
        full_webhook_url = f"{webhook_url}{webhook_path}"

        logger.info(f"Starting in WEBHOOK mode on port {port}")
        logger.info(f"Webhook URL: {full_webhook_url}")

        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=webhook_path,
            webhook_url=full_webhook_url,
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    else:
        # ── POLLING MODE ──────────────────────────────────────────
        # Good for local development or single-instance platforms
        logger.info("Starting in POLLING mode...")

        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,        # <-- Ignores old messages
            close_loop=False,
        )


if __name__ == "__main__":
    main()

