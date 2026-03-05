FROM rust:1.89-slim-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /build

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /build/recipe.json recipe.json
# Cache dependencies — this layer only rebuilds when dependencies change
RUN cargo chef cook --release --all-features --recipe-path recipe.json

# Build the actual binary
COPY . .
RUN cargo build --release --all-features && \
    strip target/release/sqwale

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash sqwale

COPY --from=builder /build/target/release/sqwale /usr/local/bin/sqwale

WORKDIR /workspace

USER sqwale

ENTRYPOINT ["sqwale"]
CMD ["--help"]