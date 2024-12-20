# Introduction
YOLO (You Only Look Once) is _a series of real-time object detection systems based on convolutional neural networks. First introduced by Joseph Redmon et al. in 2015, YOLO has undergone several iterations and improvements, becoming one of the most popular object detection frameworks._ [source: Wikipedia](https://en.wikipedia.org/wiki/You_Only_Look_Once). YOLO's latest version is YOLO11, promising  better accuracy, speed and efficiency. YOLO11 supports many [modes of operation](https://docs.ultralytics.com/modes/) that range from model training, to making predictions and real-time tracking of objects.

YOLO must be integrated to EDGELESS as there are Use Cases that will take advantage of its functionality. Experiments related to building, running, and integrating YOLO with EDGELESS on NVIDIA's Jetson devices using GPU support were conducted. We successfully used pre-trained YOLO models to run AI/ML inference on images and videos.  

# Recommended model format & dependencies

According to [Jetson with Ultralytics YOLO11 guide](https://docs.ultralytics.com/guides/nvidia-jetson/) the best performance when working with Jetson devices is achieved when TensorRT models are used. In our experiments we found that inference using TensorRT models was indeed faster than using PyTorch models. In order to convert the pre-trained PyTorch YOLO models, ONNX is used as an intermediate format. This is why `onnxruntime-gpu` package is needed for exporting the models. As a result, for Jetson devices YOLO depends on
 - PyTorch
 - Torchvision
 - TensorRT and 
 - ONNX-libraries for model conversion

Ultralytics provides pre-built docker images for each JetPack version of Jetson boards, which contain the correct dependencies for GPU support. In order to install jetpack5 variant, run:
```
t=ultralytics/ultralytics:latest-jetson-jetpack5
sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
```
We experimented building our own images to see differences in size and build times and the following text summarizes our efforts and proposes the best solution(s).

# Building a Docker image with YOLO

## Base image with conda environment
Unfortunately conda's channels do not contain a PyTorch package that targets Jetson's 64-bit architecture for CUDA 11. It was found that when resolving dependencies, libmamba (the dependency solver library used by conda) installed CPU versions of PyTorch only. As mentioned in the [PyTorch doc](../doc/pytorch.md), PyTorch must be installed using a CUDA precompiled wheel file. In addition the `pytorch-gpu` package [targets CUDA 12 only](https://anaconda.org/conda-forge/pytorch-gpu/files?sort=basename&sort_order=desc).

Consequently our base image with conda cannot provide a solution.

## Possible solutions
1. Use ultralytics/ultralytics:latest-jetson-jetpack5 image
    - Uses GPU successfully, <u>works out of the box</u> 
    - Uncompressed size is 12GB
    - Provides optional dependencies for exporting Pytorch models to TensorRT

2. Build on top of [NVIDIA's L4T images](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=scoreDESC&query=l4t&page=&pageSize=), install PyTorch via precompiled wheel files, build torchvision from source (required), install TensorRT if needed and then install YOLO. Instructions and pytorch .whl files can be found [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).
    - l4t-base lacks basic needed libraries like cuBLAS and cuFFT
    - l4t-cuda images (both runtime and devel) do not contain cuDNN
    - l4t-tensorrt-runtime image lacks `nvcc`, the CUDA compiler driver, and as a result torchvision cannot be built from source.
    - l4-tensorrt-devel image:
        - takes 9.5GB of space initially
        - building torchvision from source takes around 10min
        - Installing YOLO Ultralytics also slows down the process as tensorflow and many other dependencies are needed
        - onnxruntime must also be installed separately from the proper wheel file e.g. `pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl`
    - Use l4t-pytorch image from NVIDIA 
        - contains torchvision
        - also slow process because YOLO installs many dependencies
        - 11.7 GB uncompressed, so no image-size benefits.

3. Use [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) and build an image with the required dependencies. 
    - `jetson-containers` allows creating images with custom dependencies using Docker layers.
    - Compatibility to the underlying Jetson board is ensured by the `autotag` script.
    - Dependencies are organized in packages (e.g. Numba, PyTorch).
    - Packages are essentially Dockerfiles, sometimes accompanied by test.py files.
    - The `build` script works by adding one layer over another.
    - Building starts from base images, like l4t-jetpack in the case of Jetpack 5.
    - The Dockerfiles install dependencies via apt, or pip; sometimes precompiled binaries (whl) are downloaded.
    - These Dockerfiles have a header which includes metadata like name, group, test files and dependencies of the package.
    - Each time a layer is added, a test can be run. Also all tests run after the build script adds the final layer.

    `jetson-containers` documentation can be found [here](../doc/jetson-containers.md).

## Verdict

Because of the complex dependencies of YOLO and the lengthy build times, it is best to either use the provided [ultralytics:jetson-jetpack images](https://github.com/ultralytics/ultralytics/tree/main/docker) or use [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) if the function image requires extra dependencies too.

# Running AI inference on EDGELESS node with GPU support

Follow the instructions provided in the [EDGELESS specific notes](./edgeless.md) file. However, you need to use the appropriate YOLO jetpack image according to the underlying Jetson architecture, e.g.:

  `FROM ultralytics/ultralytics:latest-jetson-jetpack5`
