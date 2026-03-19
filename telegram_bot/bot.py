import os
import re
import json
import logging
import base64
import urllib.parse
import asyncio
import threading
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GROQ_API_KEY   = os.environ["GROQ_API_KEY"]

GROQ_CHAT_URL    = "https://api.groq.com/openai/v1/chat/completions"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_HEADERS     = {"Authorization": f"Bearer {GROQ_API_KEY}"}

TEXT_MODEL   = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
AUDIO_MODEL  = "whisper-large-v3"
QC_BASE      = "https://d1loi7eremk1cu.cloudfront.net/"

# ── Prompts ───────────────────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = (
    "Eres un asistente amigable en Telegram, experto en zapatillas y ropa streetwear (reps/réplicas). "
    "Respondes SIEMPRE en español, de forma natural, breve y con buena vibra. NO incluyas ningún tag especial. "
    "Puedes hablar de sneakers, ropa, precios, tallas, combinar outfits, o cualquier otro tema. "
    "Si el usuario pregunta cómo buscar productos, díle que simplemente lo pida natural: "
    "'quiero ver unas nike dunk' o 'buscame yeezy 350'."
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
    "SOLO devuelve el JSON, nada más."
)

# ── Memory ────────────────────────────────────────────────────────────────────
history: dict[int, list] = {}

def get_history(uid: int) -> list:
    if uid not in history:
        history[uid] = []
    return history[uid]

def add_msg(uid: int, role: str, content):
    h = get_history(uid)
    h.append({"role": role, "content": content})
    if len(h) > 20:
        history[uid] = h[-20:]

# ── Smart Search Engine ──────────────────────────────────────────────────────
ALIASES = [
    (r'\baf[ -]?1s?\b',         'air force 1'),
    (r'\byzy\b',                'yeezy'),
    (r'\bnb\b',                 'new balance'),
    (r'\bj[ -]?1\b',            'jordan 1'),
    (r'\bj[ -]?4\b',            'jordan 4'),
    (r'\bj[ -]?11\b',           'jordan 11'),
    (r'\bdunk(s?)\b',           'nike dunk'),
    (r'\btech[ -]?fleece\b',    'nike tech fleece'),
    (r'\baj[ -]?(\d+)\b',       'air jordan \\1'),
    (r'\bnmds?\b',              'adidas nmd'),
    (r'\bultra ?boost\b',       'adidas ultraboost'),
    (r'\bpanda(s?)\b',          'nike dunk panda'),
    (r'\b550s?\b',              'new balance 550'),
    (r'\b990s?\b',              'new balance 990'),
    (r'\b2002r\b',              'new balance 2002r'),
    (r'\bsb\b',                 'nike sb'),
    (r'\bts\b',                 'travis scott'),
    (r'\boff ?white\b',         'off white'),
    (r'\bow\b',                 'off white'),
    (r'\bfoam ?runner\b',       'yeezy foam runner'),
    (r'\bslide(s?)\b',          'yeezy slide'),
    (r'\btf\b',                 'nike tech fleece'),
    (r'\bgazelle(s?)\b',        'adidas gazelle'),
    (r'\bsamba(s?)\b',          'adidas samba'),
    (r'\bspezial\b',            'adidas spezial'),
    (r'\bstussy\b',             'stussy'),
    (r'\bessentials?\b',        'fear of god essentials'),
    (r'\bfog\b',                'fear of god'),
    (r'\btrapstar\b',           'trapstar'),
    (r'\bcorteiz\b',            'corteiz'),
]

# Stopwords que no ayudan en búsqueda
STOPWORDS = {'de', 'la', 'el', 'las', 'los', 'un', 'una', 'unos', 'unas', 'me',
             'te', 'se', 'en', 'con', 'por', 'para', 'que', 'del', 'al', 'y',
             'o', 'a', 'es', 'son', 'hay', 'tiene', 'ver', 'quiero', 'dame',
             'muestra', 'enseña', 'busca', 'buscame', 'encuentra', 'the', 'of'}

def normalize(q: str) -> str:
    """Normaliza el texto: minúsculas, aplica aliases, limpia caracteres raros."""
    q = q.lower().strip()
    q = re.sub(r'[^\w\s]', ' ', q)  # Quitar puntuación
    for pat, rep in ALIASES:
        q = re.sub(pat, rep, q)
    return ' '.join(q.split())  # Normalizar espacios

def generate_variants(query: str) -> list:
    """Genera variantes inteligentes del query para búsqueda más amplia."""
    norm = normalize(query)
    words = [w for w in norm.split() if w not in STOPWORDS and len(w) > 1]
    
    variants = set()
    full = ' '.join(words)
    if full:
        variants.add(full)
    
    # Si tiene más de 2 palabras, probar subconjuntos
    if len(words) > 2:
        # Marca + modelo (primera + última)
        variants.add(f"{words[0]} {words[-1]}")
        # Sin la primera palabra (por si es una marca genérica)
        variants.add(' '.join(words[1:]))
        # Sin la última palabra
        variants.add(' '.join(words[:-1]))
    
    # Si tiene exactamente 2 palabras, probar cada una sola también
    if len(words) == 2:
        for w in words:
            if len(w) > 3:
                variants.add(w)
    
    return list(variants) if variants else [query.lower().strip()]

def clean_title(t: str) -> str:
    """Limpia tags HTML del título."""
    return re.sub(r'<[^>]+>', '', t).strip()

def fuzzy_match(word: str, title: str) -> float:
    """Matching difuso para encontrar palabras similares (p.ej errores de ortografía)."""
    title_words = title.split()
    if word in title:
        return 1.0
    best = 0.0
    for tw in title_words:
        ratio = SequenceMatcher(None, word, tw).ratio()
        if ratio > best:
            best = ratio
    return best

def score_product(p: dict, words: list) -> float:
    """Sistema de scoring avanzado con múltiples factores."""
    title = clean_title(p.get('title', '')).lower()
    
    # 1. Relevancia del título (matching exacto + fuzzy)
    exact_hits = sum(1 for w in words if w in title)
    fuzzy_hits = sum(fuzzy_match(w, title) for w in words if w not in title)
    total_hits = exact_hits + (fuzzy_hits * 0.6)  # Fuzzy vale 60%
    relevance = (total_hits / len(words) * 100) if words else 0
    
    # 2. Calidad del producto
    star = p.get('star', 0) or 0
    qc_count = p.get('qcImageCount', 0) or 0
    
    # Bonus por tener muchas fotos QC (más fotos = más confiable)
    qc_bonus = min(qc_count / 50, 10)  # Max 10 puntos de bonus
    
    # Bonus por buen rating
    star_bonus = star * 3 if star >= 4.0 else star * 1.5
    
    # Bonus por tener descuento activo
    discount_bonus = 5 if p.get('discountPrice') and p.get('discountPrice') < p.get('price', 0) else 0
    
    # Penalizar si star es 0 (sin reviews = riesgo)
    no_review_penalty = -10 if star == 0 else 0
    
    # Score final: relevancia tiene peso dominante
    quality = star_bonus + qc_bonus + discount_bonus + no_review_penalty
    return relevance * 1000 + quality

def build_url(query: str, page=1, page_size=40, sort='') -> str:
    params = {
        'pageNo': page,
        'pageSize': page_size,
        'productName': query,
        'sort': sort,
        'channel': 'local'
    }
    return 'https://gateway.plug4.me/goods-service/spu/page?' + urllib.parse.urlencode(params)

def parse_product(p: dict) -> dict:
    """Extrae toda la información útil de un producto, incluyendo TODAS las fotos QC."""
    price = p.get('discountPrice') or p.get('price', 0)
    orig  = p.get('price', 0) if p.get('discountPrice') else None
    wid   = p.get('channelItemNo', '')
    
    # Extraer TODAS las fotos QC disponibles (no solo la primera)
    qc_images = []
    for group_name, group in p.get('qcImageVoGroupMap', {}).items():
        if group:
            for img in group[:5]:  # Max 5 por grupo
                url = img.get('imageUrl', '') or img.get('webpUrl', '')
                if url:
                    full_url = url if url.startswith('http') else QC_BASE + url
                    qc_images.append(full_url)
    
    # Calcular ahorro si hay descuento
    savings = None
    if orig and price and orig > price:
        savings = round(((orig - price) / orig) * 100)
    
    return {
        'title':     clean_title(p.get('title', '')),
        'price':     price,
        'orig':      orig,
        'savings':   savings,
        'image':     p.get('mainImage', ''),
        'star':      p.get('star', 0),
        'qc_count':  p.get('qcImageCount', 0),
        'qc_images': qc_images,
        'qc_img':    qc_images[0] if qc_images else None,
        'shop':      p.get('shopName', ''),
        'is_new':    p.get('newFlag') == 1,
        'wid':       wid,
        'source':    p.get('sourceUrl', ''),
        'url_plug':  f'https://plug4.me/item/{wid}?channel=weidian',
        'url_hipo':  f'https://hipobuy.com/product/2/{wid}',
        'url_hipos': f'https://hipobuys.com/index/detail?id={wid}&channel=weidian',
    }

async def do_search(query: str, sort='', top_n=5) -> list:
    """Búsqueda inteligente con múltiples variantes y deduplicación."""
    variants = generate_variants(query)
    all_products = {}  # wid -> (score, product_raw)
    
    async with httpx.AsyncClient(timeout=15) as client:
        for variant in variants[:3]:  # Max 3 variantes para no saturar
            try:
                r = await client.get(build_url(variant, sort=sort))
                r.raise_for_status()
                data = r.json()
                if data.get('success'):
                    words = [w for w in normalize(query).split() if w not in STOPWORDS and len(w) > 1]
                    for p in data['result'].get('records', []):
                        wid = p.get('channelItemNo', '')
                        if not wid:
                            continue
                        s = score_product(p, words)
                        # Mantener el mejor score si hay duplicados
                        if wid not in all_products or s > all_products[wid][0]:
                            all_products[wid] = (s, p)
            except Exception as e:
                logger.warning(f"Error buscando variante '{variant}': {e}")
                continue
    
    if not all_products:
        return []
    
    # Ordenar por score y devolver top_n
    sorted_products = sorted(all_products.values(), key=lambda x: -x[0])
    
    # Filtrar solo relevantes (score > 0), o devolver todos si ninguno es relevante
    relevant = [(s, p) for s, p in sorted_products if s > 0]
    pool = relevant if relevant else sorted_products
    
    return [parse_product(p) for _, p in pool[:top_n]]

# ── Product Display ──────────────────────────────────────────────────────────
def fmt_product(p: dict, idx: int) -> str:
    """Formato bonito del producto para Telegram."""
    title = p['title'][:60] + ('…' if len(p['title']) > 60 else '')
    
    # Precio con ahorro
    price = f"💰 ${p['price']:.2f}"
    if p.get('savings') and p['orig']:
        price += f"  ~~${p['orig']:.2f}~~  🔥 *-{p['savings']}%*"
    elif p['orig'] and p['orig'] > p['price']:
        price += f"  ~~${p['orig']:.2f}~~"
    
    # Rating con estrellas visuales
    star = p.get('star', 0)
    if star:
        filled = '⭐' * int(star)
        stars = f"{filled} {star:.1f}"
    else:
        stars = "🔘 Sin reseñas"
    
    # QC info
    qc = f"📸 {p['qc_count']} fotos QC" if p['qc_count'] else "📸 Sin QC aún"
    
    # Badges
    badges = []
    if p.get('is_new'):
        badges.append("🆕 NUEVO")
    if p.get('savings') and p['savings'] >= 20:
        badges.append("🔥 OFERTA")
    if p.get('star', 0) >= 4.5:
        badges.append("👑 TOP RATED")
    if p.get('qc_count', 0) >= 100:
        badges.append("✅ VERIFICADO")
    badge_str = ' '.join(badges)
    
    lines = [f"*{idx}. {title}*"]
    if badge_str:
        lines.append(badge_str)
    lines.append(price)
    lines.append(f"{stars} | {qc}")
    lines.append(f"🏪 {p['shop']}")
    
    return '\n'.join(lines)

def product_kb(p: dict) -> InlineKeyboardMarkup:
    """Teclado de botones con links de compra y fotos QC."""
    row1 = []
    row2 = [
        InlineKeyboardButton("🔗 Plug4.me", url=p['url_plug']),
        InlineKeyboardButton("🛒 Hipobuy",  url=p['url_hipo']),
        InlineKeyboardButton("📋 Hipobuys", url=p['url_hipos']),
    ]
    
    # Si hay fotos QC, ponerlas en la primera fila
    if p.get('qc_img'):
        row1.append(InlineKeyboardButton("📸 Ver QC Real", url=p['qc_img']))
    if len(p.get('qc_images', [])) > 1:
        row1.append(InlineKeyboardButton(f"🖼 +{len(p['qc_images'])-1} fotos QC", url=p['qc_images'][1]))
    
    keyboard = []
    if row1:
        keyboard.append(row1)
    keyboard.append(row2)
    
    return InlineKeyboardMarkup(keyboard)

async def send_products(update: Update, products: list, header: str):
    await update.message.reply_text(header, parse_mode="Markdown")
    for i, p in enumerate(products, 1):
        caption = fmt_product(p, i)
        try:
            await update.message.reply_photo(
                photo=p['image'], caption=caption,
                parse_mode="Markdown", reply_markup=product_kb(p)
            )
        except Exception:
            await update.message.reply_text(
                caption, parse_mode="Markdown", reply_markup=product_kb(p)
            )

# ── Groq AI ──────────────────────────────────────────────────────────────────
async def groq_chat(uid: int, user_msg) -> str:
    add_msg(uid, "user", user_msg)
    model    = VISION_MODEL if isinstance(user_msg, list) else TEXT_MODEL
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + get_history(uid)
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_CHAT_URL, headers=GROQ_HEADERS,
                              json={"model": model, "messages": messages, "max_tokens": 1024})
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
    add_msg(uid, "assistant", reply)
    return reply

async def groq_classify(text: str) -> dict:
    """Clasificador de intención ultra-preciso."""
    messages = [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user",   "content": text},
    ]
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(GROQ_CHAT_URL, headers=GROQ_HEADERS,
                              json={"model": TEXT_MODEL, "messages": messages,
                                    "max_tokens": 80, "temperature": 0})
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
    try:
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        return json.loads(match.group()) if match else {"intent": "chat"}
    except Exception:
        return {"intent": "chat"}

async def groq_transcribe(audio: bytes) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_WHISPER_URL, headers=GROQ_HEADERS,
                              files={"file": ("audio.ogg", audio, "audio/ogg")},
                              data={"model": AUDIO_MODEL, "response_format": "json"})
        r.raise_for_status()
        return r.json()["text"]

async def groq_describe(image: bytes, mime="image/jpeg") -> str:
    b64     = base64.b64encode(image).decode()
    content = [
        {"type": "text", "text": "Describe esta imagen detalladamente. Si hay texto, transcríbelo."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_CHAT_URL, headers=GROQ_HEADERS,
                              json={"model": VISION_MODEL,
                                    "messages": [{"role": "user", "content": content}],
                                    "max_tokens": 1024})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def dl(bot, file_id: str) -> bytes:
    f = await bot.get_file(file_id)
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f.file_path)
        r.raise_for_status()
        return r.content

# ── Commands ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "amigo"
    await update.message.reply_text(
        f"👋 ¡Hola *{name}*! Soy tu asistente de reps 🤖👟\n\n"
        f"*¿Qué puedo hacer?*\n"
        f"� Buscar productos — solo pídelo natural\n"
        f"💬 Charlar de lo que quieras\n"
        f"🎤 Entender tus audios\n"
        f"� Analizar fotos\n\n"
        f"*Ejemplos:*\n"
        f"• _\"quiero ver unas jordan 4\"_\n"
        f"• _\"búscame yeezy slides\"_\n"
        f"• _\"qué tal están las dunk panda\"_\n\n"
        f"¡Escríbeme lo que necesites! 😎",
        parse_mode="Markdown"
    )

async def cmd_buscar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(ctx.args)
    if not query:
        await update.message.reply_text(
            "✍️ Dime qué quieres buscar\nEj: _nike dunk panda_", parse_mode="Markdown"
        )
        return
    await update.message.chat.send_action("typing")
    try:
        products = await do_search(query, top_n=5)
        if not products:
            await update.message.reply_text(
                f"😕 No encontré resultados para *{query}*.\n"
                f"💡 Prueba con otro nombre o en inglés.",
                parse_mode="Markdown"
            )
            return
        await send_products(update, products,
            f"✅ *Top {len(products)} resultados para \"{query}\":*\n"
            f"🏷 _Ordenados por relevancia y calidad_"
        )
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        await update.message.reply_text("❌ Error al buscar. Intenta de nuevo.")

async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(ctx.args)
    if not query:
        await update.message.reply_text("✍️ Ej: _yeezy 350_", parse_mode="Markdown")
        return
    await update.message.chat.send_action("typing")
    try:
        products = await do_search(query, top_n=1)
        if not products:
            await update.message.reply_text("😕 No encontré resultados.")
            return
        await send_products(update, products, f"🏆 *Mejor resultado para \"{query}\":*")
    except Exception as e:
        logger.error(e)
        await update.message.reply_text("❌ Error al buscar.")

async def cmd_precio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(ctx.args)
    if not query:
        await update.message.reply_text("✍️ Ej: _jordan 1_", parse_mode="Markdown")
        return
    await update.message.chat.send_action("typing")
    try:
        products = await do_search(query, sort='price_asc', top_n=5)
        if not products:
            await update.message.reply_text("😕 No encontré resultados.")
            return
        await send_products(update, products, f"💸 *Más baratos para \"{query}\":*")
    except Exception as e:
        logger.error(e)
        await update.message.reply_text("❌ Error al buscar.")

async def cmd_qc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Comando para ver todas las fotos QC del mejor resultado."""
    query = ' '.join(ctx.args)
    if not query:
        await update.message.reply_text("✍️ Ej: `/qc nike dunk`", parse_mode="Markdown")
        return
    await update.message.chat.send_action("typing")
    try:
        products = await do_search(query, top_n=1)
        if not products:
            await update.message.reply_text("� No encontré resultados.")
            return
        p = products[0]
        qc_imgs = p.get('qc_images', [])
        if not qc_imgs:
            await update.message.reply_text(
                f"📸 *{p['title'][:50]}* no tiene fotos QC aún.\n"
                f"Puedes ver el producto aquí: [Plug4.me]({p['url_plug']})",
                parse_mode="Markdown"
            )
            return
        await update.message.reply_text(
            f"� *Fotos QC de:* {p['title'][:50]}\n"
            f"⭐ {p['star']:.1f} | 🏪 {p['shop']}\n"
            f"_Mostrando {min(len(qc_imgs), 8)} de {p['qc_count']} fotos:_",
            parse_mode="Markdown"
        )
        for img_url in qc_imgs[:8]:
            try:
                await update.message.reply_photo(photo=img_url)
            except Exception:
                await update.message.reply_text(f"� [Ver foto]({img_url})", parse_mode="Markdown")
    except Exception as e:
        logger.error(e)
        await update.message.reply_text("❌ Error al buscar fotos QC.")

# ── Message handlers ──────────────────────────────────────────────────────────
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    text = update.message.text
    await update.message.chat.send_action("typing")

    try:
        # Paso 1: clasificar intención
        classification = await groq_classify(text)
        intent = classification.get("intent", "chat")
        query  = classification.get("query", "").strip()

        if intent == "search" and len(query) > 2:
            # Es búsqueda — buscar y enviar productos
            ctx.args = query.split()
            await cmd_buscar(update, ctx)
        else:
            # Charla normal
            reply = await groq_chat(uid, text)
            await update.message.reply_text(reply, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error en handle_text: {e}")
        await update.message.reply_text("😔 Algo salió mal, inténtalo de nuevo.")

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid   = update.effective_user.id
    await update.message.chat.send_action("typing")
    voice = update.message.voice or update.message.audio
    audio = await dl(ctx.bot, voice.file_id)
    transcript = await groq_transcribe(audio)
    reply = await groq_chat(uid, f"[Transcripción de audio]: {transcript}")
    await update.message.reply_text(
        f"🎤 *Transcripción:* _{transcript}_\n\n{reply}", parse_mode="Markdown"
    )

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid   = update.effective_user.id
    await update.message.chat.send_action("typing")
    image = await dl(ctx.bot, update.message.photo[-1].file_id)
    desc  = await groq_describe(image)
    cap   = update.message.caption or ""
    prompt = f"[Imagen]: {desc}" + (f"\n[Caption]: {cap}" if cap else "")
    reply  = await groq_chat(uid, prompt)
    await update.message.reply_text(reply, parse_mode="Markdown")

async def handle_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid   = update.effective_user.id
    await update.message.chat.send_action("typing")
    video = update.message.video
    if video.thumbnail:
        image = await dl(ctx.bot, video.thumbnail.file_id)
        desc  = await groq_describe(image)
    else:
        desc = "un video sin miniatura disponible"
    cap    = update.message.caption or ""
    prompt = f"[Video frame]: {desc}" + (f"\n[Caption]: {cap}" if cap else "")
    reply  = await groq_chat(uid, prompt)
    await update.message.reply_text(reply, parse_mode="Markdown")

# ── Main ──────────────────────────────────────────────────────────────────────
class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is alive!")
    def log_message(self, format, *args):
        pass  # Silenciar logs del servidor falso

def keep_alive():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), DummyHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

def main():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        
    keep_alive()
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_start))
    app.add_handler(CommandHandler("buscar", cmd_buscar))
    app.add_handler(CommandHandler("top",    cmd_top))
    app.add_handler(CommandHandler("precio", cmd_precio))
    app.add_handler(CommandHandler("qc",     cmd_qc))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO,   handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO,                   handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO,                   handle_video))
    logger.info("🤖 Bot iniciado!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
