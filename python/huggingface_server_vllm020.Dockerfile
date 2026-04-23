ARG VLLM_IMAGE=vllm/vllm-openai:nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52
ARG WORKSPACE_DIR=/kserve-workspace

FROM ${VLLM_IMAGE} AS base

USER root
WORKDIR ${WORKSPACE_DIR}

RUN command -v uv >/dev/null || ( \
        (command -v curl >/dev/null || (apt-get update && apt-get install -y curl)) \
        && curl -LsSf https://astral.sh/uv/install.sh | sh \
        && ln -sf /root/.local/bin/uv /usr/local/bin/uv \
    )

COPY kserve/pyproject.toml kserve/uv.lock kserve/
RUN --mount=type=cache,target=/root/.cache/uv cd kserve \
    && uv pip install --system . --no-cache --no-deps
COPY kserve kserve
RUN --mount=type=cache,target=/root/.cache/uv cd kserve \
    && uv pip install --system . --no-cache

COPY storage storage
RUN --mount=type=cache,target=/root/.cache/uv cd storage \
    && uv pip install --system . --no-cache

COPY huggingfaceserver huggingfaceserver
RUN --mount=type=cache,target=/root/.cache/uv cd huggingfaceserver \
    && uv pip install --system . --no-cache

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
