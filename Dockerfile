FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY .streamlit/ .streamlit/
COPY mycographx/ mycographx/
COPY app.py ./

EXPOSE 8501

HEALTHCHECK CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"]

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
