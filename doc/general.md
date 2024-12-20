# Run code that accesses the GPU from a Docker container
Quickjump:
[Prerequisites](#prerequisites) | [GPU Enabled docker test](#gpu-enabled-docker-test) | [Experimentation with Rust technologies](#experimentation-with-rust-technologies) | [Miscellaneous
](#miscellaneous) 

In addition, in the following documentation files, more information is provided regarding a specific topic:
* [YOLO](./YOLO.md)
* [Pytorch](./pytorch.md)
* [Jetson containers](./jetson-containers.md)
* [Minimize image size](minimal-size_image.md)

## How to Jetson Q&A

Q: Which Linux for Tegra (L4T) do I have?

A: Run `jtop`. The top line of this utility gives us the Model as well as the Linux version (35.2.1 in our case) 

Q: Can i see more info?

A: Run `tegrastats`

Q: Which cuda version do I have?

A: Run `ls /usr/local/ | grep -i "cuda"`. The default location for cuda is this. Another way is to run `nvcc --version`


## Prerequisites
* One Jetson Board, e.g. NVIDIA Jetson Xavier NX or NVIDIA AGX Orin 

* `jtop` ([installation](https://jetsonhacks.com/2023/02/07/jtop-the-ultimate-tool-for-monitoring-nvidia-jetson-devices/)) package for GPU monitoring
* `nvidia-container-toolkit`

[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```
sudo apt-get install -y nvidia-container-toolkit
```
* The 2 container images discussed below

### Creating conda environment (Not required, but may be handy to have available in the NVIDIA Jetson device)

We will use cuda from inside the containers, so a global conda installation is not necessary. But it is useful in order to debug that everything is working with cuda, before messing with the containers.

* Anaconda installation https://forums.developer.nvidia.com/t/anaconda-for-jetson-nano/74286/3

Create environment `conda create -n cuda-python python=3.11`

```
conda activate cuda-python
```
```
pip install cuda-python
```

### Necessary Docker Images

```
docker pull nvidia/cuda:11.4.3-devel-ubuntu20.04
```
```
docker pull nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```

## Testing GPU access outside of containers 
(Update: the following is probably not necessary to be done. The `nvidia-container-toolkit` handles the communication with the device)

```
https://askubuntu.com/questions/997557/libcuda-so-1-not-found-despite-installing-cuda
```

Download Archiconda (conda for arm)
```
wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
```
```
bash Archiconda3-0.2.3-Linux-aarch64.sh
```
Create environment
```
conda create -n cuda-python python=3.11
```
Confirm environment creation
```
conda env list
```
Activate created environment
```
conda activate cuda-python
```
Install necessary packages
```
conda install numpy
```
```
conda install numba
```


## GPU Enabled docker test
### Numba


```
import sys
import numpy as np
from numba import cuda, float32

@cuda.jit
def matrix_multiply(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = 0
        for k in range(A.shape[1]):
            C[i, j] += A[i, k] * B[k, j]

if len(sys.argv) < 3:
    print("Please provide values for N and M as command-line arguments.")
    print("Example: python script.py 1024 1024")
    sys.exit(1)

N = int(sys.argv[1])
M = int(sys.argv[2])

# Generate random matrices
A = np.random.rand(N, M).astype(np.float32)
B = np.random.rand(M, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Configure grid and block dimensions
threadsperblock = (16, 16)
blockspergrid_x = (N + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (N + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Allocate memory on the GPU
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.device_array((N, N), dtype=np.float32)

# Launch the kernel
matrix_multiply[blockspergrid, threadsperblock](d_A, d_B, d_C)

# Copy the result back to the CPU
d_C.copy_to_host(C)

print(C)
```

Start container
```
sudo docker run -it --runtime=nvidia nvidia/cuda:11.4.3-devel-ubuntu20.04 bash
```

Setup container
```
apt-get update
apt-get install -y python3
apt-get install -y wget
apt-get -y install pip
pip install numpy
pip install numba
```

Run numba script
```
python3 matrix_mul_gpu.py 1000 1000
```
### PyTorch
[PyTorch versions for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

```
wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl -O torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
```

```
apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
```

```
pip3 install Cython
```

```
pip3 install numpy torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
```

### [Pytorch for Jetson](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

```
sudo docker pull nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```

```
sudo docker run -it --rm --runtime nvidia --network host nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```

Test that torch can be correctly imported and execute simple commands

```
python3 -c "print('Importing torch...')
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))
c = a + b
print('Tensor c = ' + str(c))
print('Importing torchvision...')
import torchvision
print(torchvision.__version__)"
```

---
## (NOT RELATED TO CUDA)
## Experimentation with Rust technologies

### Rust env

```
conda create -n rust-env python=3.11
```

```
conda activate rust-en
```

```
pip install maturin
```

### Rust installation

```
curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh
```

### [Calling Rust from Python code](http://saidvandeklundert.net/learn/2021-11-18-calling-rust-from-python-using-pyo3/)

```
mkdir rs_from_python
cd rs_from_python
```

The following contents are needed. Place them into the `Cargo.toml` file
```
[package]
name = "multiply_num"
version = "0.1.0"
edition = "2018"

[lib]
name = "rust"
# "cdylib" is necessary to produce a shared library for Python to import from.
#
# Downstream Rust code (including code in `bin/`, `examples/`, and `tests/`) will not be able
# to `use string_sum;` unless the "rlib" or "lib" crate type is also included, e.g.:
# crate-type = ["cdylib", "rlib"]
crate-type = ["cdylib"]

[dependencies.pyo3]
version = "0.15.0"
features = ["extension-module"]
```

```
touch Cargo.toml
```
... and copy the contents with your preferred way


### Miscellaneous
#### Some useful links saved when debugging containers that did not access the GPU:

[Kernel version:](https://itsfoss.com/find-which-kernel-version-is-running-in-ubuntu/)
```
$ uname -r
5.10.104-tegra
```

[Compute capability:](https://forums.developer.nvidia.com/t/what-compute-capability-of-jetson-xavier-nx-gpu-is/146241/3)
```
$ /usr/local/cuda-11.4/samples/1_Utilities/deviceQuery/deviceQuery | grep -i capability
  CUDA Capability Major/Minor version number:    7.2
```

[CUDA version:](https://forums.developer.nvidia.com/t/manually-installing-cuda-11-0-2-on-jetson-xavier-nx-help/191909)
```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Sun_Oct_23_22:16:07_PDT_2022
Cuda compilation tools, release 11.4, V11.4.315
Build cuda_11.4.r11.4/compiler.31964100_0
```

[dGPU or iGPU?](https://docs.nvidia.com/clara-holoscan/archive/clara-holoscan-0.1.0/dgpu_setup.html)

```
$ nvgpuswitch.py query
iGPU (nvidia-l4t-cuda, 35.2.1-20230124153320)
```

[NVIDIA Docker:](https://github.com/NVIDIA/nvidia-docker)

```
sudo nvidia-ctk runtime configure
```
```
$ sudo nvidia-ctk runtime configure
INFO[0000] Loading docker config from /etc/docker/daemon.json
INFO[0000] Successfully loaded config
INFO[0000] Flushing docker config to /etc/docker/daemon.json
INFO[0000] Successfully flushed config
INFO[0000] Wrote updated config to /etc/docker/daemon.json
INFO[0000] It is recommended that the docker daemon be restarted.
```

```
sudo systemctl restart docker
```

maturin expects the rust source code to be at `src/lib.rs`

```
maturin develop
```
