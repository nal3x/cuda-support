# Building a reduced-size docker function image with GPU support for [EDGELESS](https://edgeless-project.eu/)
Quickjump
[Intro](#intro) | [Function](#function) | [Dependencies and conda environment](#dependencies-and-conda-environment) | [Benefits of the approach](#benefits-of-the-approach) | [Notes](#notes)

## Intro
The following example draws from past experiments conducted and shows how a developer of the EDGELESS system can minimize the size of the docker function image using conda. The [base CUDA image](../scripts/build-cuda-image-base.sh) was successfully produced as a result of the effort.

## Function
The example demonstrates a GPU-accelerated matrix multiplication using Numba for CUDA. Numba is a JIT compiler that translates Python into fast machine code. Numba enables CUDA GPU programming and natively supports NumPy's arrays and dtypes.
The Python code for the function is defined in [`matrix_multiply_gpu.py`](../test/matrix_multiply_gpu.py)

Matrix initialization occurs on the CPU, while multiplication is handled by the GPU. 
Care must be taken because the @cuda.jit decorator doesn't directly work on instance methods (methods defined inside a class and accessed via self). As a result, trying to call `self.matrix_multiply[threads_per_block, blocks_per_grid](A, B, C)` leads to an error. 
One solution is to define the actual CUDA multiplication function (`matrix_multiply`) as an inner method of an instance method (`multiply_matrices`).

## Dependencies and conda environment
Conda was used to install the required dependencies for both the function and the docker image for EDGELESS. Conda automatically manages conflicts between dependencies, choosing compatible versions of them. All requirements are bundled in the `environment.yml` file, which conda uses to create its isolated environment, named 'cuda-python'.
```yml
name: cuda-python
channels:
 - conda-forge
dependencies:
 - cudatoolkit=11.4
 - numpy
 - numba
 - black
 - click
 - grpcio
 - grpcio-tools
 - mypy_extensions
 - packaging
 - pathspec
 - platformdirs
 - protobuf
 - tomli
 - typing_extensions
```
Numba needs the CUDA Toolkit, and, specifically for CUDA 11, the `cudatoolkit` [is required](https://numba.readthedocs.io/en/stable/cuda/overview.html#software). As described in the previous section, NumPy is needed for array manipulation.
All other dependencies are needed for integration with the EDGELESS system. 

## Dockerfile
In an effort to overcome the limited internal storage constraints of Jetson devices, the following approaches were adopted:
* Use of a minimal-size base image. The [NVIDIA Linux4Tegra (L4T) base image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base) includes essential packages and uses the NVIDIA Container Runtime to mount platform specific hardware dependencies into the container from the underlying host, making it compatible with various Jetson devices. After r34.1 release, the l4t-base does not bring CUDA from the host.
* Use of conda to install and automatically handle dependencies and conflicts, for example to install cudatoolkit which is needed by Numba.
```Dockerfile
FROM nvcr.io/nvidia/l4t-base:r36.2.0                                                                    

ENV CONDA_DIR=/opt/conda                                                                                

RUN wget  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && \
	/bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
	rm /tmp/miniconda.sh
	
#Add conda to PATH                                                                                      
ENV PATH="$CONDA_DIR/bin:$PATH"                                                                         

#Set working directory within the container 
WORKDIR /app                                                                                                           

#Copy dependencies (environemnt.yml) and app source code to image                                                                                                           
COPY . . 

# Use bash as the shell for running commands 
SHELL ["/bin/bash", "-c"]

# Create conda environment from environment.yml
RUN conda update conda && conda env create -f /app/environment.yml -v && conda clean -all -y 

ENV LOG_LEVEL=INFO
ENV PORT=7101
ENV MAX_WORKERS=10
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
EXPOSE $PORT

#ENTRYPOINT is given in shell form in order to invoke a bash shell
ENTRYPOINT conda run -n cuda-python python src/function.py --log-level $LOG_LEVEL --port $PORT --max-workers $MAX_WORKERS
```
# Benefits of the approach
The benefits of using conda include:
  * Conda comes with Python in general. In the example, we used Miniconda, a free minimal installer for conda. It is a small bootstrap version of Anaconda that includes only conda, Python, the packages they both depend on, and a small number of other useful packages (like pip, zlib, and a few others).
  * Isolated environments. Each environemnt can have its own dependencies, Python version, and paths. By using multiple environment YAML files, the EDGELESS function developer can make use of different environments for his/her functions, if needed. 
  * As a package manager, conda can handle both Python and non-Python dependencies (e.g., CUDA, BLAS libraries). Conda's packages may include precompiled binaries and, in contrast to pip, offers comprehensive conflict resolution for dependencies.

# Notes
* In our experiments we tried to use `nvcr.io/nvidia/l4t-cuda:11.4.19-runtime` but it does not include python and nvvm, leading to a `numba.cuda.cudadrv.error.NvvmSupportError: libNVVM cannot be found`. The NVVM (NVIDIA CUDA Compiler) is generally part of the full CUDA toolkit and is often included in developer images, not in the minimal runtime images like l4t-cuda-runtime.
* Using `nvcr.io/nvidia/l4t-cuda:11.4.19-devel` resulted in a (working) image with a size of 5.89GB. The described approach, which used `nvcr.io/nvidia/l4t-base` as the base image and cudatoolkit  provided from conda, resulted in an image of just **3.6GB**. 
* The dependencies in conda's `environment.yml` file for the EDGELESS docker image are the same as those in `requirements.txt`, the file used by pip, found in [edgeless-project](https://github.com/edgeless-project/runtime-python) github repo. In that file all dependencies were declared with their versions. Removing the versions in `environment.yml` leaded to a significant reduction of the time needed to build the image, because it was easier for conda's libmamba (alternative dependency resolver and package manager) to manage dependency conflicts. The integration of the python-runtime with the EDGELESS system was unaffected and after casting the function event to the EDGELESS node through the edgeless CLI, there was evidence that the computation was handled by the GPU.