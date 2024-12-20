# dusty-nv / jetson-containers

## Intro

[dusty-nv / jetson-containers](https://github.com/dusty-nv/jetson-containers) is a modular container build system that provides software packages for NVIDIA Jetson boards. The software ensures that the combination of the chosen packages are compatible with JetPack/L4T and CUDA versions of the underlying Jetson device. Developers can combine packages to build containers with dependencies tailored to their needs. 

Some of the provided packages (essentially images) include pytorch, tensorflow, ollama, llama-vision, numba, ROS, mamba etc. NVIDIA's l4t images are also included and some of them are used as base images to build containers with custom dependencies. The full list of packages can be found [here](https://github.com/dusty-nv/jetson-containers/tree/master/packages).

## Basic scripts
The basic scripts are:
 - `autotag`: finds a container image that's compatible with the version of JetPack/L4T - either locally, pulled from a registry, or by building it.
 - `build`: serves as a proxy launcher for `jetson_containers/build.py`
   - Uses build-container to build one multi-stage container from  a chain of packages
   - For reference, the method’s signature, as found in `jetson_containers/container.py` is:  
   ```
   build_container(args.name, args.packages, args.base, args.build_flags, args.build_args, args.simulate, args.skip_tests, args.test_only, args.push, args.no_github_api, args.skip_packages)
   ```
 - `run`: launches `docker run` with some added defaults (like --runtime nvidia, mounted /data cache and devices)

   `jetson-containers run $(autotag numba)` will automatically pull or build & run a compatible container image with numba and its dependencies. 

### `autotag`
An example of running `autotag` shows that `jetson-containers` correctly identifies the Jetpack, L4T, and CUDA versions:
```
$ echo $(autotag pytorch)

Namespace(disable=[''], output='/tmp/autotag', packages=['pytorch'], prefer=['local', 'registry', 'build'], quiet=False, user='dustynv', verbose=False)
-- L4T_VERSION=35.2.1  JETPACK_VERSION=5.1  CUDA_VERSION=11.4
-- Finding compatible container image for ['pytorch']
…
Found compatible container dustynv/pytorch:1.11-r35.3.1 (2023-12-14, 5.4GB) - would you like to pull it?
```
The search order for the images is: 
 - Local images (found under docker images)
 - Pulled from registry (by default hub.docker.com/u/dustynv)
 - Build from source (it'll ask for confirmation first) 

 ### `build`
 #### How build works
 In order to understand how the `build` script works, one can examine the output of a dry run using the `--simulate` argument.  

```
$ jetson-containers build --simulate --name custom pytorch torchvision
```
```
Namespace(base='', build_args='', build_flags='', list_packages=False, logs='', multiple=False, name='custom', no_github_api=False, package_dirs=[''], packages=['pytorch', 'torchvision'], push='', show_packages=False, simulate=True, skip_errors=False, skip_packages=[''], skip_tests=[''], test_only=[''], verbose=False)
-- L4T_VERSION=35.2.1
-- JETPACK_VERSION=5.1
-- CUDA_VERSION=11.4
-- PYTHON_VERSION=3.8
-- LSB_RELEASE=20.04 (focal)
-- Package comfyui has missing dependencies, disabling...  ("couldn't find package:  torchao")
    ...
-- Building containers  ['build-essential', 'pip_cache:cu114', 'cuda', 'cudnn', 'python', 'numpy', 'cmake', 'onnx', 'pytorch:2.2', 'torchvision']
-- Building container custom:r35.2.1-build-essential

DOCKER_BUILDKIT=0 docker build --network=host --tag custom:r35.2.1-build-essential \
--file /shared/jetson-containers/packages/build/build-essential/Dockerfile \
--build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r35.2.1 \
/shared/jetson-containers/packages/build/build-essential \
2>&1 | tee /shared/jetson-containers/logs/20241121_011204/build/custom_r35.2.1-build-essential.txt; exit ${PIPESTATUS[0]}

-- Building container custom:r35.2.1-pip_cache_cu114

DOCKER_BUILDKIT=0 docker build --network=host --tag custom:r35.2.1-pip_cache_cu114 \
--file /shared/jetson-containers/packages/cuda/cuda/Dockerfile.pip \
--build-arg BASE_IMAGE=custom:r35.2.1-build-essential \
--build-arg TAR_INDEX_URL="http://jetson.webredirect.org:8000/jp5/cu114" \
--build-arg PIP_INDEX_REPO="http://jetson.webredirect.org/jp5/cu114" \
```
So after identifying the underlying software stack, `build` chooses to use the l4t-jetpack:r35.2.1 image as base and on top of it, it adds the build-essential, pip_cache:cu114, cuda, cudnn layers etc, one by one, until all pytorch and torchvision dependencies are satisfied. 

#### Tests
Another excerpt from the build dry run shows that after building some of the layers, some tests are run. At the end of building, all tests from all layers are run again.  
```
-- Building container custom:r35.2.1-cuda

DOCKER_BUILDKIT=0 docker build --network=host --tag custom:r35.2.1-cuda \
   ...
-- Testing container custom:r35.2.1-cuda (cuda:11.4/test.sh)

docker run -t --rm --runtime=nvidia --network=host \
--volume /shared/jetson-containers/packages/cuda/cuda:/test \
--volume /shared/jetson-containers/data:/data \
--workdir /test \
custom:r35.2.1-cuda \
```

In order to reduce build time some or all of the tests can be skipped:
```
$ jetson-containers build --skip-tests=numpy,onnx pytorch # skip the testing of numpy and onnx packages when building pytorch
```
```
$ jetson-containers build --skip-tests=all pytorch # skip all tests
```
```
$ jetson-containers build --skip-tests=intermediate pytorch # only run tests at the end of the container build
```

## Packages
The contents of a package include:
 - Dockerfile and (optionally)
 - Configuration scripts 
    - the Python config scripts are executed at the beggining of a build, and can dynamically set build parameters based on the environment
 - Build scripts
 - Test scripts

 Dockerfile example from the numba package:
 ```dockerfile
 #---
# name: numba
# group: cuda
# depends: [cuda, numpy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
      	llvm-dev \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean \
	&& llvm-config --version
    
# https://github.com/numba/llvmlite/issues/621#issuecomment-737100914
#RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 1
#RUN llvm-config --version

RUN pip3 install --no-cache-dir --verbose numba && \
	pip3 show numba && python3 -c 'import numba; print(numba.__version__)'

 ``` 
 The above example shows that Dockerfiles contain metadata at the beginning, which help the `build` script to find package dependencies and use the test files. 

 ## Parametrizing builds

 ### Changing the base image
 Changing the base image allows to add packages to an existing container:
```
jetson-containers build --base=my_container:latest --name=my_container:pytorch pytorch  
```
will add pytorch to mycontainer

The default base image depends on the Jetpack version:

| JetPack version | Def. base image |
|---|---|
| 4 | l4t-base | 
| 5 | l4t-jetpack |
| 6 | ubuntu:22.04 |

For example a developer could add PyTorch on top of the EDGELESS runtime-python image.

### Parametrizing builds with environment variables

The jetson_containers python module and some core packages (through config.py files) expose ENV variables that can be used to parametrize builds. A developer can build an image for a Jetson with different specs and/or request a different version of a package. The following table lists these variables:

| Exposed by jetson_containers | Exposed by core packages (in config files)|
|---|---|
| L4T_VERSION | CUDA_VERSION | 
| JETPACK_VERSION | CUDNN_VERSION |
| PYTHON_VERSION | TENSORRT_VERSION |
| CUDA_VERSION | PYTHON_VERSION |
| CUDA_ARCHITECTURES | PYTORCH_VERSION |
| SYSTEM_ARCH |  |
| LSB_RELEASE |  |
| LSB_CODENAME |  |

#### Examples
Tests were run on a Jetson with the following specs:
```
-- L4T_VERSION=35.2.1
-- JETPACK_VERSION=5.1
-- CUDA_VERSION=11.4
-- PYTHON_VERSION=3.8
-- LSB_RELEASE=20.04 (focal)
```

`$ jetson-containers build pytorch torchvision` installs pytorch 2.2 and torchvision 0.17.2

`$ PYTORCH_VERSION=2.0 jetson-containers build pytorch torchvision` installs pytorch 2.0, torchvision 0.15.1, downgrading torchvision's version to preserve compatibility

`$ L4T_VERSION=36.2 CUDA_VERSION=12.4 jetson-containers build torch torchvision` will not work because of unsupported Python version. The output shows the modified environment variables that are set. Observe that Jetpack is upgraded.

```
-- L4T_VERSION=36.2
-- JETPACK_VERSION=6.0
-- CUDA_VERSION=12.4
-- PYTHON_VERSION=3.8
-- LSB_RELEASE=20.04 (focal)
```

`$ L4T_VERSION=36.2 PYTHON_VERSION=3.10 jetson-containers build torch torchvision` does not work because CUDA >= 12.4 is needed. Output:
```
-- L4T_VERSION=36.2
-- JETPACK_VERSION=6.0
-- CUDA_VERSION=11.4
-- PYTHON_VERSION=3.10
-- LSB_RELEASE=20.04 (focal)
```

`$ L4T_VERSION=36.2 PYTHON_VERSION=3.10 CUDA_VERSION=12.4 jetson-containers build torch torchvision` successfully builds the image with PyTorch 2.4 and torchvision 0.19.1. 

`$ L4T_VERSION=36.2 PYTHON_VERSION=3.10 CUDA_VERSION=12.4 PYTORCH_VERSION=2.5 jetson-containers build torch torchvision` will successfully install PyTorch 2.5 & torchvision 0.20.0

`$ JETPACK_VERSION=6.0 CUDA_VERSION=12.4 PYTHON_VERSION=3.10 PYTORCH_VERSION=2.5 jetson-containers build torch torchvision` does not work because neither JetPack nor L4T version is modified.

The official documentation states:
*“The dependencies are also able to specify with requires which versions of L4T, CUDA, and Python they need, so changing the CUDA version has cascading effects downstream and will also change the default version of … PyTorch (similar to how changing the PyTorch version also changes the default version of torchvision and torchaudio).”*

 *"The reverse also occurs in the other direction, for example changing the TensorRT version will change the default version of CUDA (unless you explicitly specify it otherwise).*"
 
 The previous statement could not be verified by our tests.