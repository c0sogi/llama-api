### Dockerfile for Python 3.11.4 & CUDA 12.1.1 & Ubuntu 22.04
### Approximately 7 ~ 10 minutes to build

# 필요한 CUDA 버전을 선택합니다.
ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}
ENV PYTHON_VERSION="3.11.4"
ENV PYTHON_VERSION_SHORT="3.11"
ENV HOST 0.0.0.0
ENV PORT=8000

# 필요한 파일들을 복사합니다.
COPY requirements.txt /tmp/requirements.txt

# Python 설치를 위한 종속성들을 설치하고, Python을 설치하고 설정합니다.
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
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/*

# 작업 디렉토리를 설정하고, 서버를 실행합니다.
WORKDIR /app
ENTRYPOINT [ "python3", "-m", "main" "--port", "${PORT}" ]
