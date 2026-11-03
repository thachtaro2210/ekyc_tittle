FROM python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
EXPOSE 8000
# One worker: each worker holds its own copy of all models, and requests are serialized by a lock anyway.
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "1", "--threads", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
