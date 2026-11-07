# kurag-txtai: txtai's ANN behind a small HTTP API, the vector index of kurag with VECTOR_BACKEND=txtai.
#   docker build -t kurag-txtai backends/txtai
# The packages are installed in a Debian 13 Python 3.13 image and copied onto distroless Python 3.13 (Debian 13),
# so the runtime has no shell, pip or package manager. Both stages must have the same Python version.
FROM python:3.13-slim-trixie AS build
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --target /site -r /tmp/requirements.txt && mkdir /data

FROM gcr.io/distroless/python3-debian13:nonroot
# The nonroot user of the distroless image, stated so scanners see it
USER 65532:65532
COPY --from=build /site /site
COPY --from=build --chown=65532:65532 /data /data
COPY server.py /app/server.py
ENV PYTHONPATH=/site PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data LISTEN=0.0.0.0:9670
WORKDIR /app
VOLUME /data
EXPOSE 9670
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/python3", "server.py"]
