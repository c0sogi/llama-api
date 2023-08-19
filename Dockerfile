# Select the required CUDA version.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder

ENV PYTHON_VERSION="3.11.4" \
    PYTHON_VERSION_SHORT="3.11" \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_DOCKER_ARCH=all

# Install the necessary applications, and then install Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget libsqlite3-dev gcc ocl-icd-opencl-dev opencl-headers clinfo libclblast-dev libopenblas-dev \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz -O /tmp/Python-${PYTHON_VERSION}.tgz \
    && tar -xvf /tmp/Python-${PYTHON_VERSION}.tgz -C /tmp \
    && cd /tmp/Python-${PYTHON_VERSION} \
    && ./configure && make && make install \
    && python3 -m pip install --upgrade pip --no-cache-dir \
    && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/* \
    && update-alternatives --install /usr/bin/python python /usr/local/bin/python${PYTHON_VERSION_SHORT} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python${PYTHON_VERSION_SHORT} 1 \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && nvcc --version

# Copy the necessary files.
COPY llama_api /app/llama_api
COPY pyproject.toml requirements.txt main.py model_definitions.py /app/

# Install the necessary Python packages(Dependencies).
RUN cd /app && python3 -m llama_api.server.app_settings --install-pkgs --force-cuda --no-cache-dir

# Set the working directory and start the server.
STOPSIGNAL SIGINT
WORKDIR /app
ENTRYPOINT [ "python3", "-m", "main", "--port", "${PORT}" ]
