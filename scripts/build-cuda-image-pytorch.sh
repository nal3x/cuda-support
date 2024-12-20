#!/bin/bash

source ./global.sh

VERSION="1.0.0" 

docker build --no-cache \
    -t "${DOCKER_REGISTRY}"/edgeless/jetson-cuda-image-pytorch:v"${VERSION}" \
    -f ../Dockerfiles/pytorch-conda_image/l4t-pytorch.Dockerfile \
    ../Dockerfiles/pytorch-conda_image/
