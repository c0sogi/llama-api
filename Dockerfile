### Dockerfile for Python 3.11.4 & CUDA 12.1.1 & Ubuntu 22.04
### Approximately 5 ~ 10 minutes to build

# Select the required CUDA version.
ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}
ENV PYTHON_VERSION="3.11.4"
ENV PYTHON_VERSION_SHORT="3.11"
ENV HOST 0.0.0.0
ENV PORT=8000

# Copy the necessary files.
COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY llama_api /app/llama_api

# Install the necessary applications, and then install Python.
# Then, install the necessary Python packages(Dependencies).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    wget \
    git \
    libsqlite3-dev\
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz -O /tmp/Python-${PYTHON_VERSION}.tgz \
    && tar -xvf /tmp/Python-${PYTHON_VERSION}.tgz -C /tmp \
    && cd /tmp/Python-${PYTHON_VERSION} \
    && ./configure \
    && make \
    && make install \
    && update-alternatives --install /usr/bin/python python /usr/local/bin/python${PYTHON_VERSION_SHORT} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python${PYTHON_VERSION_SHORT} 1 \
    && python3 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* \
    && cd /app \
    && python3 -m llama_api.server.app_settings --force-cuda --install-pkgs

# Set the working directory and start the server.
WORKDIR /app
ENTRYPOINT [ "python3", "-m", "main", "--port", "${PORT}" ]