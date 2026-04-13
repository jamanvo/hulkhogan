FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-root

COPY . .

EXPOSE 8010

CMD ["uvicorn", "config.asgi:application", "--host",  "0.0.0.0", "--port", "8010"]