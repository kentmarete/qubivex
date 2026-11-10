# syntax=docker/dockerfile:1
#
# QUBIVEX server image: `qubivex serve` on Linux x86-64 or arm64, CPU backend.
# Build:  docker build -t qubivex .
# Run:    see docs/deployment/linux-deployment.md
#
# The binary picks its SIMD path at run time (AVX2 and FMA on x86-64, NEON on arm64), so
# one image per architecture runs on any CPU of that architecture.

# Any Rust from the workspace's minimum (1.88) onwards builds it.
ARG RUST_VERSION=1.88
ARG DEBIAN_VERSION=bookworm

FROM rust:${RUST_VERSION}-${DEBIAN_VERSION} AS build
WORKDIR /src
COPY . .
# The cache mounts keep the registry and the build directory between builds; the binary
# is copied out of the mount because the mount is not part of the layer.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p qubivex-cli \
    && cp target/release/qubivex /usr/local/bin/qubivex

FROM debian:${DEBIAN_VERSION}-slim
# No extra packages: the binary links only glibc, and its TLS roots are compiled in.
RUN useradd --system --uid 10001 --create-home --home-dir /var/lib/qubivex qubivex \
    && mkdir -p /models \
    && chown qubivex:qubivex /models
COPY --from=build /usr/local/bin/qubivex /usr/local/bin/qubivex
# The tiny generated models, for a smoke test with no download:
#   docker run --rm -e QUBIVEX_API_KEY=sk-test -p 8080:8080 qubivex \
#       serve --listen 0.0.0.0:8080 --dir /usr/share/qubivex/models --model olmoe-tiny
COPY fixtures/models /usr/share/qubivex/models

USER qubivex
WORKDIR /var/lib/qubivex
# Mount a volume of models here; `models pull` installs into it too.
ENV QUBIVEX_MODELS_DIRS=/models
VOLUME ["/models"]
EXPOSE 8080

# /health needs no key. The slim image has no curl, so bash opens the socket itself.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/8080 && printf 'GET /health HTTP/1.0\\r\\n\\r\\n' >&3 && head -n 1 <&3 | grep -q ' 200 '"]

# `docker stop` sends SIGTERM, which starts the same drain as Ctrl-C. Give it longer than
# --shutdown-grace (30 s by default): docker stop --time 40.
ENTRYPOINT ["qubivex"]
# Listening beyond loopback is refused without an API key: set QUBIVEX_API_KEY, or pass
# --allow-unauthenticated when something in front of the container authenticates.
CMD ["serve", "--listen", "0.0.0.0:8080"]
