# EDGELESS-GPU
This repository contains both [general notes](./doc/general.md) related to GPUs, NVIDIA, and Jetson devices gathered during research activities on offloading computation to accelerators (i.e., GPUs), as well as [EDGELESS specific notes](./doc/edgeless.md) to demonstrate GPU utilization within the EDGELESS system. 


## Project-Structure
```
.
├── doc             # Directory containing documentation
├── Dockerfiles     # Dockerfiles created for EDGELESS
├── scripts         # Auxiliary scripts
└── test            # Files used during tests
```

## Usage

In the [scripts/](./scripts/) directory, there are multiple scripts that can be used to build the provided [Dockerfiles](./Dockerfiles/).

The general purpose of this repository is to provide Jetson images to EDGELESS nodes, enabling direct utilization of popular AI/ML libraries and frameworks, which abstract the use of CUDA, to offload computations to the GPU.

1. Before proceeding, start by reading the [general notes](./doc/general.md) to gain a basic understanding.

2. Then, navigate to the [scripts/](./scripts/) directory and run one of the `build-cuda-image-<version>.sh` scripts to build the provided images.

3. Next, refer to the [EDGELESS specific notes](./doc/edgeless.md) to understand how the generated image can be utilized within the EDGELESS platform.

## Provided images
At this point, multiple images are provided, each offering distinct benefits. 

1. `<DOCKER_REGISTRY>/edgeless/jetson-cuda-image-ml`

    Internally, it uses the [NVIDIA L4T ML](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml) as the base image, which provides the following packages:

    ```
    - TensorFlow 2.11.0
    - PyTorch v2.0.0
    - torchvision v0.15.1
    - torchaudio v2.0.1
    - onnx 1.13.1
    - onnxruntime 1.16.0
    - optimum 1.8.8
    - CuPy 13.0.0
    - numpy 1.23.5
    - numba 0.56.4
    - PyCUDA 2022.2
    - OpenCV 4.5.0 (with CUDA)
    - pandas 2.0.1
    - scipy 1.10.0
    - scikit-learn 1.2.2
    - diffusers 0.17.1
    - transformers 4.30.2
    - xformers 0.0.20
    - JupyterLab 3.6.3
    ```

    This, however, results in an image size of approximately 19 GB and an initialization time of nearly 50 seconds.

2. `<DOCKER_REGISTRY>/edgeless/jetson-cuda-image-base`

    This image uses [NVIDIA L4T BASE](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base) internally and installs `conda`. Its advantage is a smaller size, less than 4 GB, and a significantly faster initialization time.

    By updating the [enviroment.yml](./Dockerfiles/base-conda_image/environment.yml) file, this image can accommodate the necessary packages, as it does not support the same frameworks out of the box.

3. `<DOCKER_REGISTRY>/edgeless/jetson-cuda-image-pytorch`
    This image uses [NVIDIA Jetpack](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack) internally and it has support for `PyTorch`. Read [this](./doc/pytorch.md) if you need in depth info regarding the `Pytorch` support.

## Test
In the [test/](./test/) directory, there are Python files used in the EDGELESS platform to demonstrate GPU utilization from virtualized environments.