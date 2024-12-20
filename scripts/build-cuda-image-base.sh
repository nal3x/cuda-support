#!/bin/bash

source ./global.sh

VER="1.0.0"

docker build --no-cache \
    -t "${DOCKER_REGISTRY}"/edgeless/jetson-cuda-image-base:v"${VER}" \
    -f ../Dockerfiles/base-conda_image/l4t-base.Dockerfile \
    ../Dockerfiles/base-conda_image/
