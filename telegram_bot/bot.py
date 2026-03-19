import os
import re
import logging
import base64
import urllib.parse
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

SYSTEM_PROMPT = (
    "Eres un asistente amigable en Telegram, experto en zapatillas y ropa streetwear (reps). "
    "Respondes siempre en español de forma útil y concisa. "
    "Si notas explícita o implícitamente que el usuario quiere buscar zapatillas, un producto o ropa, "
    "DEBES incluir en tu respuesta esta etiqueta exacta: <SEARCH>término_de_búsqueda</SEARCH>. "
    "Por ejemplo: '¡Claro! Te busco eso enseguida. <SEARCH>nike air force 1</SEARCH>'. "
    "La respuesta debe ser conversacional, y la etiqueta disparará automáticamente la búsqueda."
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

# ── Search ────────────────────────────────────────────────────────────────────
ALIASES = [
    (r'\baf[ -]?1\b',    'air force 1'),
    (r'\byzy\b',         'yeezy'),
    (r'\bnb\b',          'new balance'),
    (r'\bj[ -]?1\b',     'jordan 1'),
    (r'\bdunk\b',        'nike dunk'),
]

def normalize(q: str) -> str:
    q = q.lower().strip()
    for pat, rep in ALIASES:
        q = re.sub(pat, rep, q)
    return q

def clean_title(t: str) -> str:
    return re.sub(r'<[^>]+>', '', t).strip()

def build_url(query: str, page_size=40, sort='') -> str:
    p = {'pageNo': 1, 'pageSize': page_size, 'productName': query, 'sort': sort, 'channel': 'local'}
    return 'https://gateway.plug4.me/goods-service/spu/page?' + urllib.parse.urlencode(p)

def score(p: dict, words: list) -> float:
    title = clean_title(p.get('title', '')).lower()
    hits  = sum(1 for w in words if w in title)
    rel   = round((hits / len(words)) * 100) if words else 0
    qual  = (p.get('star', 0) * 10) + (p.get('qcImageCount', 0) * 0.1)
    return rel * 1000 + qual

def parse_product(p: dict) -> dict:
    price  = p.get('discountPrice') or p.get('price', 0)
    orig   = p.get('price', 0) if p.get('discountPrice') else None
    wid    = p.get('channelItemNo', '')
    qc_img = None
    for group in p.get('qcImageVoGroupMap', {}).values():
        if group:
            url    = group[0].get('imageUrl', '')
            qc_img = url if url.startswith('http') else QC_BASE + url
            break
    return {
        'title':     clean_title(p.get('title', '')),
        'price':     price,
        'orig':      orig,
        'image':     p.get('mainImage', ''),
        'star':      p.get('star', 0),
        'qc_count':  p.get('qcImageCount', 0),
        'qc_img':    qc_img,
        'shop':      p.get('shopName', ''),
        'is_new':    p.get('newFlag') == 1,
        'url_plug':  f'https://plug4.me/item/{wid}?channel=weidian',
        'url_hipo':  f'https://hipobuy.com/product/2/{wid}',
        'url_hipos': f'https://hipobuys.com/index/detail?id={wid}&channel=weidian',
    }

async def do_search(query: str, sort='', top_n=5) -> list:
    norm  = normalize(query)
    words = [w for w in norm.split() if len(w) > 1]
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(build_url(norm, sort=sort))
        r.raise_for_status()
        data = r.json()
    if not data.get('success'):
        return []
    records  = data['result']['records']
    scored   = sorted([(score(p, words), p) for p in records], key=lambda x: -x[0])
    relevant = [(s, p) for s, p in scored if s > 0] or scored
    return [parse_product(p) for _, p in relevant[:top_n]]

def fmt_product(p: dict, idx: int) -> str:
    title = p['title'][:60] + ('…' if len(p['title']) > 60 else '')
    price = f"💰 ${p['price']:.2f}"
    if p['orig'] and p['orig'] > p['price']:
        price += f" ~~(${p['orig']:.2f})~~"
    stars = f"⭐ {p['star']:.1f}" if p['star'] else "⭐ Sin rating"
    new   = " 🆕" if p['is_new'] else ""
    return f"*{idx}. {title}*{new}\n{price}\n{stars} | 📸 {p['qc_count']} fotos QC\n🏪 {p['shop']}"

def product_kb(p: dict) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton("🔗 Plug4.me", url=p['url_plug']),
        InlineKeyboardButton("🛒 Hipobuy",  url=p['url_hipo']),
        InlineKeyboardButton("📋 Hipobuys", url=p['url_hipos']),
    ]
    if p.get('qc_img'):
        buttons.insert(0, InlineKeyboardButton("📸 Foto QC Real", url=p['qc_img']))
        keyboard = [[buttons[0], buttons[1]], [buttons[2], buttons[3]]]
    else:
        keyboard = [[buttons[0], buttons[1], buttons[2]]]
        
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

# ── Groq ──────────────────────────────────────────────────────────────────────
async def groq_chat(uid: int, user_msg) -> str:
    add_msg(uid, "user", user_msg)
    model    = VISION_MODEL if isinstance(user_msg, list) else TEXT_MODEL
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + get_history(uid)
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_CHAT_URL, headers=GROQ_HEADERS,
                              json={"model": model, "messages": messages, "max_tokens": 1024})
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
    add_msg(uid, "assistant", reply)
    return reply

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
        f"👋 ¡Hola *{name}*! Soy tu asistente AI 🤖\n\n"
        f"Puedo hacer de todo:\n"
        f"💬 Responder cualquier pregunta con IA\n"
        f"🎤 Transcribir audios y notas de voz\n"
        f"🖼 Analizar imágenes\n"
        f"👟 *Buscar reps* en Hipobuy / Hipobuys / Plug4.me\n\n"
        f"*Comandos de búsqueda:*\n"
        f"`/buscar nike air force 1` — top 5 resultados\n"
        f"`/top yeezy 350` — el mejor resultado\n"
        f"`/precio jordan 1` — más barato primero\n\n"
        f"O simplemente escríbeme lo que necesites 😎",
        parse_mode="Markdown"
    )

async def cmd_buscar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(ctx.args)
    if not query:
        await update.message.reply_text("✍️ Ej: `/buscar nike dunk`", parse_mode="Markdown")
        return
    await update.message.chat.send_action("typing")
    msg = await update.message.reply_text(f"🔍 Buscando *{query}*...", parse_mode="Markdown")
    try:
        products = await do_search(query, top_n=5)
        await msg.delete()
        if not products:
            await update.message.reply_text("😕 No encontré resultados.")
            return
        await send_products(update, products, f"✅ *Top {len(products)} resultados para {query}:*")
    except Exception as e:
        logger.error(e)
        await msg.edit_text("❌ Error al buscar. Intenta de nuevo.")

async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(ctx.args)
    if not query:
        await update.message.reply_text("✍️ Ej: `/top yeezy 350`", parse_mode="Markdown")
        return
    await update.message.chat.send_action("typing")
    msg = await update.message.reply_text(f"🏆 Buscando el mejor *{query}*...", parse_mode="Markdown")
    try:
        products = await do_search(query, top_n=1)
        await msg.delete()
        if not products:
            await update.message.reply_text("😕 No encontré resultados.")
            return
        await send_products(update, products, f"🏆 *Mejor resultado para {query}:*")
    except Exception as e:
        logger.error(e)
        await msg.edit_text("❌ Error al buscar.")

async def cmd_precio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(ctx.args)
    if not query:
        await update.message.reply_text("✍️ Ej: `/precio jordan 1`", parse_mode="Markdown")
        return
    await update.message.chat.send_action("typing")
    msg = await update.message.reply_text(f"💸 Buscando más barato: *{query}*...", parse_mode="Markdown")
    try:
        products = await do_search(query, sort='price_asc', top_n=5)
        await msg.delete()
        if not products:
            await update.message.reply_text("😕 No encontré resultados.")
            return
        await send_products(update, products, f"💸 *Más baratos para {query}:*")
    except Exception as e:
        logger.error(e)
        await msg.edit_text("❌ Error al buscar.")

# ── Message handlers ──────────────────────────────────────────────────────────

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    text = update.message.text
    await update.message.chat.send_action("typing")
            
    reply = await groq_chat(uid, text)
    
    # Verificamos si la IA decidió mandar a buscar algo
    search_match = re.search(r'<SEARCH>(.*?)</SEARCH>', reply, re.IGNORECASE)
    
    if search_match:
        query = search_match.group(1).strip()
        # Removemos la etiqueta del mensaje final
        clean_reply = re.sub(r'<SEARCH>.*?</SEARCH>', '', reply, flags=re.IGNORECASE).strip()
        
        if clean_reply:
            await update.message.reply_text(clean_reply, parse_mode="Markdown")
            
        ctx.args = query.split()
        await cmd_buscar(update, ctx)
    else:
        await update.message.reply_text(reply, parse_mode="Markdown")

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
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import os

class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is alive!")

def keep_alive():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), DummyHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

def main():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        
    keep_alive()  # Inicia el servidor web falso para engañar a Render
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_start))
    app.add_handler(CommandHandler("buscar", cmd_buscar))
    app.add_handler(CommandHandler("top",    cmd_top))
    app.add_handler(CommandHandler("precio", cmd_precio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO,   handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO,                   handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO,                   handle_video))
    logger.info("🤖 Bot iniciado!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
