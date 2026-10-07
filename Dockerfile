FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf \
    /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app ./app
COPY train.py ./train.py

RUN mkdir -p \
    /app/artifacts \
    /app/data/ECHO

ENV ECHO_DATA_DIR=/app/data/ECHO
ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONUNBUFFERED=1

EXPOSE 8000


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]