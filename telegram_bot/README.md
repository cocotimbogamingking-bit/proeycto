# Telegram AI Bot — Groq + Llama

Bot de Telegram con IA usando Groq (Llama 3.3 + Llama 4 Scout + Whisper).

## Funciones
- 💬 Responde mensajes de texto con Llama 3.3-70b
- 🎤 Transcribe audios y notas de voz con Whisper
- 🖼️ Describe imágenes con Llama 4 Scout (vision)
- 🎬 Describe videos usando el thumbnail
- 🧠 Memoria de conversación por usuario

## Variables de entorno necesarias
- `TELEGRAM_TOKEN` — token de tu bot de BotFather
- `GROQ_API_KEY` — tu API key de Groq

## Deploy en Render
1. Sube este proyecto a GitHub
2. Ve a render.com → New → Background Worker
3. Conecta tu repo
4. Agrega las variables de entorno
5. Deploy
