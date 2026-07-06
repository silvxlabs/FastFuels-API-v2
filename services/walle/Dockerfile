FROM python:3.13-slim

RUN pip install uv

COPY services/lib/ /app/lib/
COPY services/walle/ /app/walle/

WORKDIR /app/walle

RUN uv sync --frozen

ENV PYTHONPATH=/app/walle

ENTRYPOINT ["uv", "run", "python", "-m", "walle"]
