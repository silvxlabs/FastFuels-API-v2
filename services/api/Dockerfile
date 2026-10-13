FROM python:3.13-slim

RUN pip install uv

COPY services/lib/ /app/lib/
COPY services/api/ /app/api/

WORKDIR /app/api

RUN uv sync --locked

ENTRYPOINT ["uv", "run", "--no-sync", "gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "0", "--preload", "api.main:app"]
