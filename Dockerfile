FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pipeline ./pipeline
# Bake a trained model into the image so every service starts with the same artefact.
RUN python -m pipeline train
EXPOSE 8000
CMD ["python", "-m", "pipeline", "api"]
