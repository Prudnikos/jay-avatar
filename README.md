# Jay Streaming Avatar — GPU Pod

Streaming MuseTalk inference сервер для Jay Personal AI.

Принимает текст по WebSocket, синтезирует речь через ElevenLabs, гонит
её через MuseTalk realtime, муксует видео + аудио в fragmented MP4
сегменты по 1.5 сек и стримит обратно клиенту через тот же WebSocket.

Браузер собирает MP4-чанки через MediaSource Extensions в один
`<video>` элемент — **аудио и видео share one media clock**, sync
физически невозможен сломаться.

## Файлы

| Файл | Что делает |
|---|---|
| `server.py` | FastAPI приложение. WebSocket handler. |
| `realtime_engine.py` | Wrapper над MuseTalk. Имеет DEMO режим как fallback. |
| `mp4_segment_muxer.py` | PyAV-based fragmented MP4 muxer. NVENC если доступен. |
| `tts.py` | ElevenLabs PCM 16kHz клиент. |
| `auth.py` | HMAC-SHA256 token validation (общий секрет с Worker-ом). |
| `prepare_avatar.py` | CLI для one-time подготовки аватара (latents). |
| `config.py` | Загрузка конфига из env. |
| `Dockerfile` | Образ для деплоя. |
| `requirements.txt` | Python зависимости. |
| `start.sh` | Entrypoint. Качает модели если нет, запускает uvicorn. |
| `DEPLOY.md` | Пошаговая инструкция деплоя на RunPod. |

## Локальный smoke test (без GPU)

В DEMO режиме можно тестировать pipeline без GPU/MuseTalk:

```bash
cd gpu-pod/
pip install fastapi uvicorn websockets httpx loguru python-dotenv pydantic numpy opencv-python-headless av imageio
export MODE=demo
export JAY_AVATAR_TOKEN_SECRET=dev-secret-1234
export ELEVENLABS_API_KEY=<your-key>
export AVATARS_ROOT=/tmp/jay-avatars
python -m uvicorn server:app --port 8000

# In a separate terminal:
curl http://localhost:8000/health
```

Для тестирования WebSocket в DEMO режиме нужен клиент-скрипт (можно
использовать widget.js указав на `ws://localhost:8000` через DevTools).

## Архитектура inference

```
text input
   │
   ▼
ElevenLabs TTS  ──────►  PCM 16kHz mono float32
                                │
                                ▼
                  ┌─────────────┴──────────────┐
                  │ for each 1.5s chunk:       │
                  │                             │
                  │  audio_chunk → Whisper-tiny  │
                  │     ↓                        │
                  │  features → UNet (cuda)      │
                  │     ↓                        │
                  │  pred_latents → VAE decode   │
                  │     ↓                        │
                  │  face crops 256x256          │
                  │     ↓                        │
                  │  paste back → 256x256 frame  │
                  │     ↓                        │
                  │  PyAV: H.264 + AAC mux      │
                  │     ↓                        │
                  │  WS.send_bytes(fmp4 chunk)   │
                  └──────────────────────────────┘
```

## Дальнейшие улучшения (Phase 2/3)

- **Streaming TTS** через ElevenLabs WebSocket API: первый PCM-чанк
  через ~200ms вместо 600-1200ms на полный mp3. Cuts ~500ms с TTFB.
- **Per-frame WebRTC** через aiortc: первый кадр через 500-700ms.
- **Avatar prewarm**: pre-load latents в VRAM при первом подключении
  business-а. Сейчас все аватары грузятся при boot pod-а.
- **Multiple concurrent sessions**: сейчас `ENGINE_LOCK` сериализует
  все запросы. Для 5+ одновременных юзеров нужны batched inference
  или несколько Pod-ов с round-robin.

## Версии и лицензии

- MuseTalk 1.5 — MIT (TMElyralab)
- whisper-tiny, ft-mse-vae, dwpose — см. их лицензии
- ElevenLabs API — proprietary, требует API ключ
- Наш код (этой папки) — для внутреннего использования Jay project
