FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    libexpat1

RUN pip install uv

COPY services/lib/ /app/lib/
COPY services/etcher/ /app/etcher/

WORKDIR /app/feature

RUN uv sync --frozen

ENV PYTHONPATH=/app/etcher
ENV FUNCTION_TARGET=process_feature_request
ENV FUNCTION_SOURCE=etcher/main.py
ENV FUNCTION_SIGNATURE_TYPE=http
ENV CPL_MACHINE_IS_GCE=YES

ENTRYPOINT ["uv", "run", "functions-framework"]
