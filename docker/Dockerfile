# kurag image: static (musl) binaries on distroless: the server (kurag) and its client (kurag-sync). Build from the
# repository root:
#   docker build -f docker/Dockerfile -t kurag .
# Only the binaries (for a release archive or the systemd unit in dist/systemd/), written to ./out:
#   docker build -f docker/Dockerfile --target bin --output type=local,dest=out .
FROM rust:1-alpine AS build
RUN apk add --no-cache musl-dev
WORKDIR /src
COPY . .
# Release profile for shipping, kept out of Cargo.toml so that local builds stay fast
ENV CARGO_PROFILE_RELEASE_LTO=true \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
    CARGO_PROFILE_RELEASE_STRIP=true
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p kurag -p kurag-sync && cp target/release/kurag target/release/kurag-sync / && mkdir /data

FROM scratch AS bin
COPY --from=build /kurag /kurag-sync /

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /kurag /kurag-sync /usr/local/bin/
COPY --from=build --chown=nonroot:nonroot /data /data
ENV LISTEN=0.0.0.0:9650 DATA_DIR=/data
VOLUME /data
EXPOSE 9650
# API_KEY and EMBED_URL are required (docs/architecture.md, "Deployment")
ENTRYPOINT ["/usr/local/bin/kurag"]
