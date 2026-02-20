FROM python:3.13-slim

RUN pip install uv

COPY services/lib/ /app/lib/
COPY services/api/ /app/api/

WORKDIR /app/api

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "gunicorn", "-w", "8", "-k", "uvicorn.workers.UvicornWorker", "api.main:app"]
