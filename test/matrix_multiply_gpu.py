import numpy as np
from numba import cuda, float32

# Modify https://github.com/edgeless-project/runtime-python/src/function_servicer.py  
# from matrix_multiply_gpu import *
                      
class MatrixOperations:
    def __init__(self):
        self.A = None
        self.B = None
        self.C = None
        self.M = None
        self.N = None

    # Modify https://github.com/edgeless-project/runtime-python/src/function_servicer.py
    # ... 
    # def Init(self, request, context):
    #     ...
    #     logger.info("init() Initialize MatrixOperation obj")
    #     self.top = MatrixOperations()

    #     logger.info("init() Initialize Matrices") 
    #     self.top.initialize_matrix(2000,2000)

    #     logger.info("init() finished!") 
    # ...
    def initialize_matrix(self, M, N):
        self.M = M
        self.N = N
        self.A = np.random.rand(self.N, self.M).astype(np.float32)
        self.B = np.random.rand(self.M, self.N).astype(np.float32)
        self.C = np.zeros((self.N, self.N), dtype=np.float32)

    # Modify https://github.com/edgeless-project/runtime-python/src/function_servicer.py
    # ...
    # def Cast(self, request, context):
    #   ...
    #   logger.info("cast() Start multiplication")
    #   self.top.multiply_matrices()
    # ...
    def multiply_matrices(self):
        # Configure grid and block dimensions
        threadsperblock = (16, 16)
        blockspergrid_x = (self.N + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (self.N + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Allocate memory on the GPU
        d_A = cuda.to_device(self.A)
        d_B = cuda.to_device(self.B)
        d_C = cuda.device_array((self.N, self.N), dtype=np.float32)

        #CUDA kernel definition for matrix multiplication
        @cuda.jit
        def matrix_multiply(A, B, C):
            (i, j) = cuda.grid(2)
            if i < C.shape[0] and j < C.shape[1]:
                C[i, j] = 0
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]

        #Launch the kernel
        matrix_multiply[blockspergrid, threadsperblock](d_A, d_B, d_C)

        # Copy the result back to the CPU
        d_C.copy_to_host(self.C)

        print(self.C)
