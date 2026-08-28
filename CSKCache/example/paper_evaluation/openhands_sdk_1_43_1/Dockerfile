FROM python:3.12-slim@sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17

ARG ADAPTER_SOURCE_SHA256

LABEL org.opencontainers.image.title="CSKCache OpenHands SDK BenchFlow adapter" \
      org.opencontainers.image.version="1.43.1" \
      io.cskcache.openhands-sdk.version="1.43.1" \
      io.cskcache.openhands-tools.version="1.43.1" \
      io.cskcache.base.digest="sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17" \
      io.cskcache.adapter.source-sha256="${ADAPTER_SOURCE_SHA256}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENHANDS_SUPPRESS_BANNER=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    CSKCACHE_SKILLS_DIR=/skills

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends ca-certificates git tmux \
    && rm -rf /var/lib/apt/lists/*

COPY official-lock-requirements.txt /tmp/official-lock-requirements.txt

RUN python -m pip install --no-cache-dir \
    -r /tmp/official-lock-requirements.txt \
    openhands-sdk==1.43.1 \
    openhands-tools==1.43.1 \
    && rm /tmp/official-lock-requirements.txt

WORKDIR /root
COPY adapter/ /opt/cskcache/openhands_sdk_1_43_1/

RUN python -c "import importlib.metadata as m; assert m.version('openhands-sdk') == '1.43.1'; assert m.version('openhands-tools') == '1.43.1'" \
    && python -m py_compile /opt/cskcache/openhands_sdk_1_43_1/*.py \
    && chmod -R a+rX /opt/cskcache/openhands_sdk_1_43_1
