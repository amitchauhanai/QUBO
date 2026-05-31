# Minimal Dockerfile for qubo FastAPI service
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENV QUBO_ALLOW_INSTALL=false
ENV QUBO_API_KEY=""
EXPOSE 8000
CMD ["python", "-m", "qubo.service"]
