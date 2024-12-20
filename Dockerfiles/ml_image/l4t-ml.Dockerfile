# XAVIER NX/AGX orin
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3  

RUN apt update -y

ARG GRPC_VER="1.60.0"
RUN pip3 install grpcio==${GRPC_VER} && pip3 install grpcio-tools==${GRPC_VER}

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN rm -rf /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python
