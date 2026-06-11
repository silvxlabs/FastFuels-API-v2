FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    libexpat1

RUN pip install uv

COPY services/lib/ /app/lib/
COPY services/uploader/ /app/uploader/

WORKDIR /app/uploader

RUN uv sync --frozen

ENV PYTHONPATH=/app/uploader
ENV FUNCTION_TARGET=process_upload
ENV FUNCTION_SOURCE=uploader/main.py
ENV FUNCTION_SIGNATURE_TYPE=cloudevent
ENV CPL_MACHINE_IS_GCE=YES

ENTRYPOINT ["uv", "run", "functions-framework"]
