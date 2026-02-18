FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    build-essential gdal-bin libgdal-dev libexpat1

RUN pip install uv

COPY services/lib/ /app/lib/
COPY services/api/ /app/api/

WORKDIR /app/api

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "gunicorn", "-w", "8", "-k", "uvicorn.workers.UvicornWorker", "api.main:app"]
