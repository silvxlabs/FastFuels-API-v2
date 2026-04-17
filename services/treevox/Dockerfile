FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    libexpat1

RUN pip install uv

COPY services/lib/ /app/lib/
COPY services/treevox/ /app/treevox/

WORKDIR /app/treevox

RUN uv sync --frozen

ENV PYTHONPATH=/app/treevox
ENV FUNCTION_TARGET=process_grid_request
ENV FUNCTION_SOURCE=treevox/main.py
ENV FUNCTION_SIGNATURE_TYPE=http
ENV CPL_MACHINE_IS_GCE=YES

ENTRYPOINT ["uv", "run", "functions-framework"]
