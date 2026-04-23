ARG VLLM_IMAGE=vllm/vllm-openai:nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52
ARG WORKSPACE_DIR=/kserve-workspace

FROM ${VLLM_IMAGE} AS base

# ARG сбрасывается после FROM — переобъявляем для доступа в RUN/chown ниже
ARG WORKSPACE_DIR=/kserve-workspace

USER root
WORKDIR ${WORKSPACE_DIR}

RUN command -v uv >/dev/null || ( \
        (command -v curl >/dev/null || (apt-get update && apt-get install -y curl)) \
        && curl -LsSf https://astral.sh/uv/install.sh | sh \
        && ln -sf /root/.local/bin/uv /usr/local/bin/uv \
    )

# Ставим kserve-специфичные Python-зависимости, которых нет в vllm-openai base.
# Список собран из python/{kserve,storage,huggingfaceserver}/pyproject.toml, исключая то,
# что уже присутствует в vllm (torch, transformers, accelerate, fastapi, uvicorn, pydantic,
# aiohttp, numpy, pandas, starlette, httpx, cryptography, prometheus_client, grpcio).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache \
        'cloudevents<2.0.0,>=1.6.2' \
        'grpc-interceptor<1.0.0,>=0.15.4' \
        'timing-asgi<1.0.0,>=0.3.0' \
        'tabulate<1.0.0,>=0.9.0' \
        'orjson<4.0.0,>=3.10.15' \
        'asgi-logger<1.0.0,>=0.1.0' \
        'kubernetes>=23.3.0' \
        'dulwich>=0.21.0' \
        'pyjwt>=2.12.0' \
        'modelscope<2.0.0,>=1.16.0'

# Все kserve-пакеты ставим с --no-deps, чтобы не переустанавливать vllm/torch/transformers/nvidia-*
# из base-image (иначе huggingfaceserver pyproject.toml даунгрейдит их до pinned версий из upstream).

COPY kserve/pyproject.toml kserve/uv.lock kserve/
RUN --mount=type=cache,target=/root/.cache/uv cd kserve \
    && uv pip install --system . --no-cache --no-deps
COPY kserve kserve
RUN --mount=type=cache,target=/root/.cache/uv cd kserve \
    && uv pip install --system . --no-cache --no-deps

COPY storage storage
RUN --mount=type=cache,target=/root/.cache/uv cd storage \
    && uv pip install --system . --no-cache --no-deps

COPY huggingfaceserver huggingfaceserver
RUN --mount=type=cache,target=/root/.cache/uv cd huggingfaceserver \
    && uv pip install --system . --no-cache --no-deps

RUN useradd kserve -m -u 1000 -d /home/kserve \
    && chown -R kserve:kserve ${WORKSPACE_DIR}

USER 1000
ENV PYTHONPATH=${WORKSPACE_DIR}/huggingfaceserver
ENV HF_HOME=/tmp/huggingface
ENV SAFETENSORS_FAST_GPU=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV VLLM_NCCL_SO_PATH=/lib/x86_64-linux-gnu/libnccl.so.2
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

ENTRYPOINT ["python3", "-m", "huggingfaceserver"]
