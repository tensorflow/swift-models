# TODO: We should have a job that creates a S4TF base image so that
# we don't have to duplicate the installation everywhere.
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

# Allows the caller to specify the toolchain to use.
ARG swift_tf_url=https://storage.googleapis.com/swift-tensorflow-artifacts/nightlies/latest/swift-tensorflow-DEVELOPMENT-ubuntu18.04.tar.gz

# Install Swift deps.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        python \
        python-dev \
        python-pip \
        python-setuptools \
        python-tk \
        python3 \
        python3-pip \
        python3-setuptools \
        clang \
        libcurl4-openssl-dev \
        libicu-dev \
        libpython-dev \
        libpython3-dev \
        libncurses5-dev \
        libxml2 \
        libblocksruntime-dev

# Configure cuda
RUN echo "/usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs" > /etc/ld.so.conf.d/cuda-9.2-stubs.conf && \
    ldconfig

# Download and extract S4TF
WORKDIR /swift-tensorflow-toolchain
RUN curl -fSsL $swift_tf_url -o swift.tar.gz \
    && mkdir usr \
    && tar -xzf swift.tar.gz --directory=usr --strip-components=1 \
    && rm swift.tar.gz
ENV PATH="/swift-tensorflow-toolchain/usr/bin:${PATH}"

WORKDIR /swift-models
COPY . .
