FROM python:3.11.3-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY main.py .

FROM base AS test_image
COPY tests tests/
RUN python -m unittest -v

FROM base AS prod_image
EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
