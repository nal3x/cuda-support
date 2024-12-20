FROM nvcr.io/nvidia/l4t-base:r35.2.1                                                                       

# ===================================================
# ===================================================
# CONDA INSTALLATION
# ------------------
ENV CONDA_DIR=/opt/conda                                                                                   

RUN wget  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && \
	/bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
	rm /tmp/miniconda.sh                                                                                   

#Add conda to PATH                                                                                        
ENV PATH="$CONDA_DIR/bin:$PATH"                                                                            

#Set working directory within the container 
WORKDIR /app                                                                                                    

#Copy dependencies (environment.yml) and app source code to image                                                                                                           
COPY environment.yml environment.yml 

# Use bash as the shell for running commands 
SHELL ["/bin/bash", "-c"]

# Create conda environment from environment.yml
RUN conda update conda && conda env create -f /app/environment.yml -v && conda clean -all -y && conda init
# ===================================================
# ===================================================

RUN rm -rf /usr/bin/python && \
ln -s /usr/bin/python3 /usr/bin/python

# Set the default conda environment name
ENV CONDA_DEFAULT_ENV=cuda-python

# Activate the conda environment by default in the shell
SHELL ["conda", "run", "-n", "cuda-python", "/bin/bash", "-c"]

# ENTRYPOINT is given in shell form in order to invoke a shell
# ENTRYPOINT conda run -n cuda-python python <python_app_path>
