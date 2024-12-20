FROM nvcr.io/nvidia/l4t-jetpack:r35.2.1
ENV CONDA_DIR=/opt/conda

RUN wget  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && \
        /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
        rm /tmp/miniconda.sh

#Add conda to PATH
ENV PATH="$CONDA_DIR/bin:$PATH"

#Set working directory within the container
WORKDIR /app

#Install OpenBlas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev \
        && \
    rm -rf /var/lib/apt/lists/*

#Copy dependencies (environment.yml) and app source code to image
COPY . .

# Use bash as the shell for running commands
SHELL ["/bin/bash", "-c"]

# Create conda environment from environment.yml
RUN conda update conda && \
    conda env create -f /app/environment-pt.yml -v && \
    conda clean -all -y && \
    conda init

# Set the default conda environment name
ENV CONDA_DEFAULT_ENV=cuda-pytorch

# Activate the conda environment by default in the shell
SHELL ["conda", "run", "-n", "cuda-pytorch", "/bin/bash", "-c"]

#Install PyTorch wheel in cuda-pytorch env
RUN python -m pip install --no-cache-dir -r requirements.txt