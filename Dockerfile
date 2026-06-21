FROM python:3.12-slim

WORKDIR /app

# 시스템 의존성 (geopandas 빌드에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY . .

# 데이터 디렉토리 권한
RUN mkdir -p data/raw data/processed data/graph models outputs/maps outputs/charts outputs/reports cache

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
