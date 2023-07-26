### Dockerfile for Python 3.11.4 & CUDA 12.1.1 & Ubuntu 22.04
### Approximately 7 ~ 10 minutes to build

# 필요한 CUDA 버전을 선택합니다.
ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}
ENV PYTHON_VERSION="3.11.4"
ENV PYTHON_VERSION_SHORT="3.11"
ENV VIRTUAL_ENV=".venv"
ENV PORT=8000

# Python 설치를 위한 종속성들을 설치합니다.
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python 소스를 다운로드 받습니다.
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz

# 압축을 풀고, 컴파일 후 설치하고, 소스를 제거합니다.
RUN tar -xvf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm Python-${PYTHON_VERSION}.tgz && \
    rm -r Python-${PYTHON_VERSION}

# 기본 Python 명령어를 설정합니다.
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python${PYTHON_VERSION_SHORT} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python${PYTHON_VERSION_SHORT} 1

# virtual environment를 만들고, 필요한 package들을 설치합니다.
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0
ENTRYPOINT [ "python3", "-m", "main" "--port", "${PORT}" ]