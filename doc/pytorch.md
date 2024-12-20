# How to install PyTorch with CUDA support
## Using the base image and PyTorch pip wheel installers for aarch64
### Problems encountered 
Experiments were carried out on Jetson boards having [Jetpack SDK 5.1](https://developer.nvidia.com/embedded/jetpack-sdk-51) installed, which includes:
  - Jetson Linux 35.2.1
  - CUDA 11.4
  - TensorRT 8.5.2 
  - cuDNN 8.6.0 
  - etc...

PyTorch requires CUDA 11.8 at least. The Compatibility Matrix for PyTorch releases can be found [here](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).
Consequently, in order to install PyTorch, we have to install CUDA toolkit 11.8 first. 
Our [base image](../Dockerfiles/l4t-base.Dockerfile) can install CUDA toolkit 11.8 from conda-forge or NVIDIA conda's channels. For example, after downgrading to python 3.11, it is possible to install pytorch 2.1 through conda by running:
  ```
  conda install pytorch==2.1.0  pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
Here, `pytorch-cuda` is a meta-package to select the proper PyTorch version. 
Unfortunately this will install a CPU version:
  ```
    >>> print(torch.cuda.is_available())
    False
    >>> torch.cuda.init()
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "/opt/conda/envs/cuda-python/lib/python3.11/site-packages/torch/cuda/__init__.py", line 265, in init
	    _lazy_init()
      File "/opt/conda/envs/cuda-python/lib/python3.11/site-packages/torch/cuda/__init__.py", line 289, in _lazy_init
	raise AssertionError("Torch not compiled with CUDA enabled")
  AssertionError: Torch not compiled with CUDA enabled
```
 In addition, the [official instructions](https://pytorch.org/) also download CPU versions of PyTorch, either through Conda:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
or Pip:
``` 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
In order to have PyTorch with <ins>GPU support</ins>, one should install PyTorch through [pre-built PyTorch pip wheel installers](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) which can also be found [here](https://developer.download.nvidia.com/compute/redist/jp/). A step-by-step procedure is described [here](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#).


### Solution

Unfortunately, use of our [l4t-base image](../Dockerfiles/l4t-base.Dockerfile) with conda and its 11.8 cuda-toolkit is impractical, as the image has many missing PyTorch dependencies like cublas, cudnn etc that have to be installed manually.

Installing the PyTorch wheel on top of [NVIDIA's L4T Jetpack image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack) with 11.8 CUDA toolkit from conda-forge will succeed:

```
(cuda-python) root@31cd910ced33:/app# python3
Python 3.8.20 | packaged by conda-forge | (default, Sep 30 2024, 17:47:05)
[GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
```

The commands that have to be run before successfully importing torch with CUDA support (or what to add in EDGELESS runtime-python's Dockerfile and requirements.txt) 
```
apt update
apt install libopenblas-dev
pip install numpy
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl
pip install $TORCH_INSTALL
```

It should be noted that trying to install the PyTorch .whl on top of [NVIDIA's L4T JetPack](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack) 5.1 image <u>without 11.8 CUDA toolkit from conda-forge</u> will not work as CUDA >=11.8 is missing (the image will include CUDA 11.4.19)
```
(cuda-python) root@f8b785393ac8:/app# pip install --no-cache $TORCH_INSTALL
ERROR: torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl is not a supported wheel on this platform.
```

This demonstrates that one can use conda's cuda-toolkit to bring an upgraded version of CUDA in a NVIDIA Docker image which includes an older version of CUDA.

## Using NVIDIA's `l4t-pytorch` docker image

[NVIDIA L4T PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch) contains PyTorch and torchvision pre-installed in a Python 3 environment. The container for [JetPack 5.1 (L4T R35.2.1)](nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3) uses 11.7GB of space uncompressed and includes
 - PyTorch v2.0.0
 - torchvision v0.14.1
 - torchaudio v0.13.1
 - Python v3.8.10

## Attempts to use pytorch-gpu from conda-forge channel
### CUDA compatibility
In general the CUDA version in a container does not have to match the CUDA version of the host machine. The key requirement is that the host machine's NVIDIA driver is compatible with the CUDA version used in the container.
### Conda's pytorch-gpu and lack of compatibility
The `pytorch-gpu` package for linux-aarch64 requires CUDA 12 (filenames starting with linux-aarch contain cuda120 and cuda126 substrings) 
as can be seen [here](https://anaconda.org/conda-forge/pytorch-gpu/files?sort=basename&sort_order=desc). The conda-forge channel [does not include](https://anaconda.org/conda-forge/cudatoolkit/files?page=3&sort=basename&sort_order=asc) `linux-aarch64/cudatoolkit-12.*` packages. Consequently, in order to have CUDA 12 in a Docker container, two possible solutions are:
  1) install [cuda-toolkit](https://anaconda.org/nvidia/cuda-toolkit) v.12 through conda provided by the NVIDIA channel
  2) use of an L4T NVIDIA Docker image which includes CUDA 12. 

  Concerning the first solution, it was found that the `pytorch-gpu` package could not be installed, even if CUDA 12.* toolkit is installed by NVIDIA channel.
  ```
  root@agx-2:/app# conda install pytorch-gpu --dry-run -c conda-forge
Channels:
 - conda-forge
 - defaults
Platform: linux-aarch64
Collecting package metadata (repodata.json): done
Solving environment: failed

LibMambaUnsatisfiableError: Encountered problems while solving:
  - package pytorch-gpu-2.1.2-cuda120py310h783a43a_203 requires pytorch 2.1.2 cuda120_py310h13b6a1d_203, but none of the providers can be installed

Could not solve for environment specs
The following packages are incompatible
├─ __cuda is requested and can be installed;
└─ pytorch-gpu is not installable because it requires
   └─ pytorch [2.1.2 cuda120_py310h13b6a1d_203|2.1.2 cuda120_py38hca0abb7_204|...|2.5.1 cuda126_py310h90e4772_201], which requires
      └─ cuda-version [>=12.0,<13 |>=12.6,<13 ], which requires
         └─ __cuda >=12 , which conflicts with any installable versions previously reported.
``` 

The second solution failed too. Trying to run [L4T CUDA images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-cuda) (runtime or devel) leads to the following error:
```
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: nvidia-container-runtime did not terminate successfully: exit status 1: time="2024-12-15T22:41:28+02:00" level=error msg="failed to create NVIDIA Container Runtime: failed to construct OCI spec modifier: requirements not met: unsatisfied condition: cuda>=12.2 (cuda=11.4)"
: unknown.
```