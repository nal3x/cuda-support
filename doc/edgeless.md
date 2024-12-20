# EDGELESS notes

## Demo
Demos that utilize the GPU from a Docker container (virtual environment) are available in the [test/](../test/) directory. The demonstrations focus on a) matrix and tensor multiplications using Numba and PyTorch respectively, and b) AI/ML inference on images or videos, to showcase the computation capabilities provided.

* [matrix_multiply_gpu.py](../test/matrix_multiply_gpu.py)

    In this example, two matrices are multiplied. The dependencies for this script are `numpy` and `numba`.

* [tensors_mul_pytorch.py](../test/tensors_mul_pytorch.py)

    In this example, two tensors are multiplied. The dependency for this script is `pytorch`.

* [yolo_inference.py](../test/yolo_inference.py)

    In this example, recognition of objects is performed on images or videos with the help of [YOLO](https://docs.ultralytics.com/), producing relevant metadata. In [this](./YOLO.md) documentation file, additional information related to YOLO can be extracted. 

    



## Reproduce Demo

1. To enable the function image to utilize CUDA on a Jetson device, Jetson’s Docker configuration `/etc/docker/daemon.json` must be modified to support NVIDIA as the default runtime.
   
    ```json
    {
        "runtimes": {
            "nvidia": {
                "args": [],
                "path": "nvidia-container-runtime"
            }
        },
        "default-runtime" : "nvidia"
    }
    ```

2. Download EDGELESS [runtime-python](https://github.com/edgeless-project/runtime-python.git) repository and set up your system as described in it.
    ```
    git clone https://github.com/edgeless-project/runtime-python.git
    ```

3. Modify [runtime-python](https://github.com/edgeless-project/runtime-python.git) Dockerfile to utilize the proper base image

    >[!Note]
    See [README.md](../README.md) to select one of them.

    **Replace**: ~~FROM python:3~~

    **With**: FROM [DOCKER_REGISTRY]/edgeless/jetson-cuda-image-[version]:[tag] 
    
    or the appropriate [ultralytics:latest-jetson-jetpack{X}](https://hub.docker.com/r/ultralytics/ultralytics/tags?name=jetson-jetpack) image.


4. Copy the corresponding [test/](../test/) Python file in the `src` directory of the [runtime-python](https://github.com/edgeless-project/runtime-python.git).

    The project tree should look like this:
    ```
    runtime-python/
    ├── CONTRIBUTORS.txt
    ├── Dockerfile
    ├── edgeless-logo-64-40.png
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── scripts
    │   └── compile-proto.sh
    └── src
        ├── function.py
        ├── function_servicer.py
        ├── matrix_multiply_gpu.py
        ├── messages_pb2_grpc.py
        ├── messages_pb2.py
        ├── messages.proto
        ├── node_cli.py
        ├── services_pb2_grpc.py
        ├── services_pb2.py
        ├── services.proto
        └── test_function_servicer.py
    ```

5. Update `runtime-python/src/function_servicer.py` to import and call the respective functions of the test file that you have selected. Comments in test files provide instructions on how to modify the `function_servicer.py` file.

6. Follow [runtime-python](https://github.com/edgeless-project/runtime-python.git) to build and run the `function-image`.

7. Follow [MVP container example](https://github.com/edgeless-project/edgeless/tree/main/examples/container) to create a `workflow.json` file that will utilize this `function-image`.


