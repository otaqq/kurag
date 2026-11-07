# kurag image: static (musl) binaries on distroless, the server (kurag) and its client (kurag-sync). Build from the
# repository root (the context needs Cargo.toml, Cargo.lock, crates/, tests/ and ui/):
#   docker build -f docker/Dockerfile -t kurag .
# or use docker/compose.yaml (README.md, "Quick start"). Only the binaries (for the systemd units in dist/systemd/),
# written to ./out:
#   docker build -f docker/Dockerfile --target bin --output type=local,dest=out .
FROM rust:1-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /src
COPY . .
# Release profile for shipping, kept out of Cargo.toml so that local builds stay fast
ENV CARGO_PROFILE_RELEASE_LTO=true \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
    CARGO_PROFILE_RELEASE_STRIP=true
# Cached registry and target: a rebuild after a small change takes seconds (the binaries are copied out of the cache)
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p kurag -p kurag-sync && \
    cp target/release/kurag target/release/kurag-sync /usr/local/bin/

FROM scratch AS bin
COPY --from=builder /usr/local/bin/kurag /usr/local/bin/kurag-sync /

# The data folder, owned by the nonroot user (distroless has no mkdir); kept apart from the builder so that the release
# image does not compile anything
FROM alpine:3 AS datadir
RUN mkdir /data

FROM gcr.io/distroless/static-debian13:nonroot AS runtime
# The nonroot user of the distroless image, stated so scanners see it
USER 65532:65532
COPY --from=datadir --chown=65532:65532 /data /data
ENV LISTEN=0.0.0.0:9650 DATA_DIR=/data
VOLUME /data
EXPOSE 9650
# distroless has no shell or curl: the binary asks its own /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s CMD ["/usr/local/bin/kurag", "health"]
# API_KEY and EMBED_URL are required (README.md, "Settings")
ENTRYPOINT ["/usr/local/bin/kurag"]

# Published images (.github/workflows/release.yml): the release binaries, built natively per architecture and put in
# bin/<arch>/ beforehand, so the multi-arch image needs no emulation and holds the same files as the release.
FROM runtime AS release
ARG TARGETARCH
COPY --chmod=755 bin/${TARGETARCH}/kurag bin/${TARGETARCH}/kurag-sync /usr/local/bin/

# Default target (last stage): built from source
FROM runtime AS production
COPY --from=builder /usr/local/bin/kurag /usr/local/bin/kurag-sync /usr/local/bin/
