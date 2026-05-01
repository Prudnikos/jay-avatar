# Jay Streaming Avatar — RunPod Deploy Guide

> **Цель**: запустить GPU Pod с MuseTalk, настроить env vars, проверить
> что health endpoint отвечает, дать его URL обратно в Cloudflare Worker.

## 0. Что нужно ДО начала

- Аккаунт на runpod.io с балансом ($10 хватит на тесты)
- ElevenLabs API key (уже есть в env Worker-а)
- Сгенерированный shared secret для токенов:
  ```bash
  openssl rand -base64 32
  # пример: VWuVHPLLAAYEEvtGRuoCKFYEMrqfdZjGVcpRO7QKHBE=
  ```
  Этот секрет ставится одинаково и на Pod, и в Worker secrets.

## 1. Создать Docker image и залить в Docker Hub

Свой реестр или Docker Hub — на выбор. Самый простой путь — Docker Hub:

```bash
cd gpu-pod/
docker build -t YOUR_DOCKERHUB_USER/jay-avatar:v1 .
docker push YOUR_DOCKERHUB_USER/jay-avatar:v1
```

**Важно**: build занимает ~10-15 минут (CUDA + MuseTalk зависимости).
Если не хочешь возиться с Docker — можно использовать готовый
RunPod template `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel` и
выполнить `pip install -r requirements.txt` + `git clone MuseTalk` уже
внутри Pod-а (см. шаг 5).

## 2. Создать Pod на RunPod

1. Идёшь в https://www.runpod.io/console/pods → **Deploy**
2. Выбираешь **GPU Cloud** → **RTX 4090** (24GB VRAM)
3. **Template**:
   - Если залил Docker Hub: **Custom** → ставишь свой image
     `YOUR_DOCKERHUB_USER/jay-avatar:v1`
   - Если без Docker: **PyTorch 2.1** template
4. **Container Disk**: 30 GB (хватит для моделей)
5. **Volume Disk**: 20 GB → mount path `/workspace/MuseTalk/models`
   (модели весят ~10GB, persistent чтобы не качать заново)
6. **Expose HTTP Ports**: `8000`
7. **Environment Variables** (тут пишешь свои значения):
   ```
   JAY_AVATAR_TOKEN_SECRET=<тот самый секрет из шага 0>
   ELEVENLABS_API_KEY=<твой ElevenLabs ключ>
   ALLOWED_ORIGINS=https://jay-platform.a2a.llc,https://jay.a2a.llc,https://payinchat.a2a.llc,https://api.a2a.llc
   MODE=musetalk            # OR demo для теста pipeline без MuseTalk
   LOG_LEVEL=INFO
   FPS=25
   SEGMENT_DURATION_S=1.5
   ```
8. **Deploy** — Pod встаёт за 30-60 секунд.

## 3. Подключиться и (если без Docker) развернуть код

Открой **Connect → Web Terminal** в RunPod консоли:

```bash
cd /workspace
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
bash download_weights.sh    # или python -c "from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk', local_dir='./models')"

mkdir -p /workspace/app
# Скопировать сюда наши файлы (server.py, realtime_engine.py и т.д.) — через scp/git/RunPod file uploader

cd /workspace/app
pip install -r requirements.txt
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

bash start.sh   # запуск
```

## 4. Подготовить аватар

Аватар = одна фотография лица, 256×256 или больше (предпочтительно
анфас, нейтральное выражение).

Загрузи фото через Web Terminal:
```bash
# В RunPod terminal:
mkdir -p /workspace/avatars/source_images
# Через RunPod file uploader загрузи jay-face.png в /workspace/avatars/source_images/

cd /workspace/app
python prepare_avatar.py --image /workspace/avatars/source_images/jay-face.png --name default
```

После скрипта в `/workspace/avatars/default/` появятся:
- `source.png` — копия исходного
- `coords.json` — bbox лица
- `latents.pt` — VAE latents (PyTorch tensor)

Перезапусти server (`Ctrl+C`, `bash start.sh`), и MuseTalk возьмёт готовый аватар.

## 5. Проверить health endpoint

В RunPod консоли возьми **Public IP / Pod URL** (выглядит как
`https://abc123-8000.proxy.runpod.net`).

Проверь:
```bash
curl https://abc123-8000.proxy.runpod.net/health
```

Должно вернуть:
```json
{
  "status": "ok",
  "mode": "musetalk",
  "uptime_s": 42,
  "avatars": ["default"],
  "mime": "video/mp4; codecs=\"avc1.42E01F,mp4a.40.2\""
}
```

Если `mode: "demo"` — значит MuseTalk не загрузился (см. логи Pod-а).
DEMO режим всё равно работает для проверки websocket/MSE pipeline.

## 6. Прописать секреты в Cloudflare Worker (production)

```powershell
cd C:\1\a2apay-worker-v2

npx wrangler secret put JAY_AVATAR_POD_URL
# Введи: https://abc123-8000.proxy.runpod.net   (без trailing /)

npx wrangler secret put JAY_AVATAR_TOKEN_SECRET
# Введи: <тот же secret из шага 0>
```

## 7. Деплой обновлённого worker-а и widget-а

```powershell
# Worker (с новыми endpoints /jay/avatar/session и /jay/avatar/health)
Copy-Item index.ts -Destination C:\1\a2apay-worker-v2\src\index.ts -Force
cd C:\1\a2apay-worker-v2
npx wrangler deploy

# Widget
Copy-Item widget.js -Destination C:\1\jay-platform\public\widget.js -Force
cd C:\1\jay-platform
npm run build
npx wrangler pages deploy dist --project-name=jay-platform
```

## 8. Smoke-test

1. Открой `https://jay-platform.a2a.llc/b/<твой-slug>` в Chrome.
2. Открой DevTools → Console. Должно появиться:
   ```
   [Jay] streaming avatar: available
   ```
3. Тапни на orb → разреши микрофон → скажи "привет".
4. Должно: видео в orb-е плавно начнётся через ~1.5-2 сек, аудио идёт
   синхронно с губами, никакого filler-болтовни не нужно.

В DevTools → Network должен быть открыт WS-конект к
`wss://abc123-8000.proxy.runpod.net/avatar/...`.

## 9. Что смотреть в логах (RunPod Pod terminal)

```
[ws abc-uuid] open slug=test-coach avatar=default
[TTS] 27840 samples (1.74s) for 78 chars
[ws abc-uuid] first byte to client at 1840ms
[ws abc-uuid] done in 2960ms (inf=820ms, mux=120ms, segs=2)
```

Если `first byte` > 3 секунд — что-то медленнее ожидаемого.
Если `mode=demo` в health — MuseTalk не загрузился, см. секцию
**Troubleshooting** ниже.

## Стоимость (контроль)

- **RTX 4090 Pod**: $0.34/hr = ~$8/день при 24/7
- Если вырубать через 10 мин неактивности (надо настроить — RunPod
  поддерживает `idle_timeout` в API): **$0.5-2/день** при 5 пилотах
- Для теста этой недели — оставь Pod включённым, считай ~$50-70 max.

## Troubleshooting

### `mode: "demo"` в health response
MuseTalk не импортировался. Зайди в Pod terminal:
```bash
cd /workspace/app
python -c "from musetalk.utils.utils import load_all_model; print('ok')"
```
Если ошибка — обычно проблема в mmcv/mmdet/mmpose версиях.
Перетестировать:
```bash
mim install "mmcv==2.0.1" --force-reinstall
mim install "mmdet==3.1.0" --force-reinstall
mim install "mmpose==1.1.0" --force-reinstall
```

### `latents.pt missing` в логах
Не запущен `prepare_avatar.py`. Запусти:
```bash
python /workspace/app/prepare_avatar.py --image /path/to/face.png --name default
```

### WS 4401 auth error
- `JAY_AVATAR_TOKEN_SECRET` не совпадает между Pod и Worker.

### WS не подключается с браузера
- CORS — добавь свой домен в `ALLOWED_ORIGINS` env var на Pod.
- Mixed content — Pod должен быть **https** (RunPod proxy URL уже https).

### Pod вырубается сам
- RunPod может вырубать Pod если кончается баланс. Проверь.
- В On-Demand Pod-ах нет idle timeout — они работают пока ты их не остановишь.
