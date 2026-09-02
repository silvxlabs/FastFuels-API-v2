FROM python:3.13-slim

RUN pip install uv

COPY services/lib/ /app/lib/
COPY services/api/ /app/api/

WORKDIR /app/api

RUN uv sync --frozen

# Trust X-Forwarded-Proto from the Cloud Run proxy so request.url_for
# builds https links. Only the platform proxy can reach the container.
ENV FORWARDED_ALLOW_IPS="*"

ENTRYPOINT ["uv", "run", "gunicorn", "-w", "8", "-k", "uvicorn.workers.UvicornWorker", "api.main:app"]
