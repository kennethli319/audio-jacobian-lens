# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ARG JLENS_MODEL_ID=openai/whisper-tiny.en
ARG JLENS_MODEL_REVISION=87c7102498dcde7456f24cfd30239ca606ed9063

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    TOKENIZERS_PARALLELISM=false \
    JLENS_MODEL=${JLENS_MODEL_ID} \
    JLENS_MODEL_REVISION=${JLENS_MODEL_REVISION} \
    PORT=7860

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/sh user \
    && mkdir -p /data /home/user/app \
    && chown -R user:user /data /home/user

WORKDIR /home/user/app

# Install locked runtime dependencies before copying frequently changed source files.
COPY --chown=user:user pyproject.toml uv.lock README.md LICENSE ./
RUN python -m pip install --no-cache-dir "uv==0.11.28" \
    && uv sync --locked --no-dev --extra audio --no-install-project

# Cache the exact public Whisper snapshot in the image. The web server loads
# the model before binding its port, so this keeps Space wake-ups independent
# of a runtime model download.
RUN HF_HOME=/data/.huggingface .venv/bin/python -c \
    "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${JLENS_MODEL_ID}', revision='${JLENS_MODEL_REVISION}', allow_patterns=['*.json', '*.safetensors', '*.txt'])" \
    && chown -R user:user /data/.huggingface

COPY --chown=user:user jlens ./jlens
COPY --chown=user:user web ./web
COPY --chown=user:user samples ./samples
COPY --chown=user:user docker/entrypoint.sh ./docker/entrypoint.sh

RUN uv sync --locked --no-dev --extra audio --no-editable \
    && install -m 0755 docker/entrypoint.sh /usr/local/bin/jlens-entrypoint

ENV HOME=/home/user \
    HF_HOME=/data/.huggingface \
    XDG_CACHE_HOME=/data/.cache \
    PATH="/home/user/app/.venv/bin:${PATH}"

USER user

EXPOSE 7860

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/jlens-entrypoint"]
