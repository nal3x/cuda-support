#!/bin/bash

source ./global.sh

GRPC_VER="1.62.1" 

docker build --no-cache \
    --build-arg="GRPC_VER=${GRPC_VER}" \
    -t "${DOCKER_REGISTRY}"/edgeless/jetson-cuda-image-ml:v"${GRPC_VER}" \
    -f ../Dockerfiles/ml_image/l4t-ml.Dockerfile \
    ../Dockerfiles/ml_image/
