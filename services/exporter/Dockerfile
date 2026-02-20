FROM python:3.13-slim

RUN pip install uv

COPY services/lib/ /app/lib/
COPY services/exporter/ /app/exporter/

WORKDIR /app/exporter

RUN uv sync --frozen

ENV PYTHONPATH=/app/exporter
ENV FUNCTION_TARGET=process_export_request
ENV FUNCTION_SOURCE=exporter/main.py
ENV FUNCTION_SIGNATURE_TYPE=http
ENV CPL_MACHINE_IS_GCE=YES

ENTRYPOINT ["uv", "run", "functions-framework"]
