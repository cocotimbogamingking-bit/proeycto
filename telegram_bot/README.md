# 🤖 RepBot — Telegram AI Bot for Reps

Bot de Telegram con IA para buscar réplicas de zapatillas y ropa streetwear.
Usa Groq (Llama 3.3 + Llama 4 Scout + Whisper) + Hipopick/Plug4me API.

## ✨ Features
- 🔍 **Búsqueda inteligente** — con aliases, variantes, fuzzy matching, scoring y deduplicación
- 💬 **Chat con IA** — Llama 3.3-70b para conversación natural
- 🎤 **Transcripción de audio** — Whisper V3 (¡también detecta si quieres buscar por voz!)
- 📷 **Análisis de imágenes** — Llama 4 Scout vision (detecta búsquedas en captions)
- 🎬 **Videos** — Análisis de thumbnails
- 🧠 **Memoria de conversación** — LRU con límite de usuarios
- ⚡ **Caché de búsqueda** — TTL de 5 minutos
- 🛡️ **Rate limiting** — Anti-spam por usuario
- 🔄 **Retry automático** — Reintenta en errores transitorios
- 📦 **Connection pooling** — HTTP client reutilizable

## 📋 Comandos
| Comando | Descripción |
|---------|-------------|
| `/start` | Mensaje de bienvenida |
| `/buscar [producto]` | Buscar productos |
| `/top [producto]` | El mejor resultado |
| `/precio [producto]` | Los más baratos |
| `/qc [producto]` | Ver fotos QC reales |
| `/reset` | Limpiar historial de conversación |

## 🔧 Variables de Entorno
| Variable | Descripción |
|----------|-------------|
| `TELEGRAM_TOKEN` | Token de BotFather |
| `GROQ_API_KEY` | API key de Groq |
| `PORT` | Puerto HTTP (default: 8080) |

## 🚀 Deploy en Render
1. Sube este proyecto a GitHub
2. Ve a render.com → New → Web Service
3. Conecta tu repo
4. Agrega las variables de entorno
5. Deploy 🚀
