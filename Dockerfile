# --- Stage 1: Build Stage ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install build essentials + libcurl (required for modern llama.cpp)
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/ggml-org/llama.cpp.git .

# Build with P40 Specific Optimizations
# - CUDA_ARCHITECTURES=61: Direct target for P40/Pascal
# - GGML_CUDA_FORCE_MMQ=ON: Essential for speed on P40
# - LLAMA_CURL=ON: Enables downloading models directly from HF
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DLLAMA_CURL=ON \
    && cmake --build build --config Release -j$(nproc)

# --- Stage 2: Runtime Stage ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install runtime dependencies (CURL runtime is needed to run the binary)
RUN apt-get update && apt-get install -y \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the compiled binaries from the builder stage
COPY --from=builder /app/build/bin/ /app/bin/

# Expose the server port
EXPOSE 8080

# Set environment variables for better P40 stability
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080

# Default command (can be overridden)
ENTRYPOINT ["/app/bin/llama-server"]
