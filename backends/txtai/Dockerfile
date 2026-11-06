# kurag-txtai: txtai's ANN behind a small HTTP API, the vector index of kurag with VECTOR_BACKEND=txtai.
#   docker build -t kurag-txtai backends/txtai
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data LISTEN=0.0.0.0:9670
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd -u 65532 -m app && mkdir /data && chown app /data
COPY server.py .
USER app
VOLUME /data
EXPOSE 9670
STOPSIGNAL SIGTERM
CMD ["python", "server.py"]
