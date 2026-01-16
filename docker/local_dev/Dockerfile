# This dockerfile is used throughout steps of the CI/CD pipeline
# It contains dependencies to run tests
FROM python:3.13

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libhdf5-dev \
    casacore-dev \
    libboost-date-time-dev \
    libboost-filesystem-dev \
    libboost-numpy-dev \
    libboost-program-options-dev \
    libboost-python-dev \
    libopenblas-dev \

    libcfitsio-dev \
    libgsl-dev \
    pybind11-dev \
    wcslib-dev \
    wsclean \
    git \
    curl
ENV MAKEFLAGS="-j 4"
RUN python -m pip install --upgrade pip
RUN python -m pip install git+https://git.astron.nl/RD/everybeam.git


RUN mkdir -p /opt/src && \
        cd /opt/src && git clone --no-checkout \
        https://gitlab.com/aroffringa/wsclean.git wsclean && \
    mkdir -p /opt/src/wsclean/build && \
    cd wsclean && git checkout ${WSCLEAN_COMMIT} && \
    git submodule update --init --recursive && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DTARGET_CPU=${TARGET_CPU} && \
    make install -j`nproc`

