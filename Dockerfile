# --- Stage 1: Build Stage (Runs on any machine) ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/ggml-org/llama.cpp.git .

# THE "SMOKING GUN" FIX:
# 1. Create the versioned symlink the linker is specifically crying for
# 2. Add it to the environment so the compiler sees it immediately
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# Build core components with P40 optimizations
# We use -Wl,-rpath-link to force the linker to accept the stubs for transitive dependencies
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build --config Release --target llama-server llama-cli -j$(nproc)

# --- Stage 2: Runtime Stage ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install runtime dependencies: 
# 1. libcurl4: for model downloading
# 2. libgomp1: for OpenMP multi-threading (FIXES YOUR ERROR)
RUN apt-get update && apt-get install -y \
    libcurl4 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled binaries from builder
COPY --from=builder /app/build/bin/ /app/bin/

# Set paths
ENV PATH="/app/bin:${PATH}"
ENV LD_LIBRARY_PATH="/app/bin:${LD_LIBRARY_PATH}"

EXPOSE 8080
ENTRYPOINT ["llama-server"]
