import torch                                                                           
import time                                                                            
import gc                                                                              

# Modify https://github.com/edgeless-project/runtime-python/src/function_servicer.py  
# from tensors_mul_pytorch import *
                                                                          
class TensorOperations:                                                                
    def __init__(self):                                                                
        self.tensor_a       = None                                                     
        self.tensor_b       = None                                                     

    # Modify https://github.com/edgeless-project/runtime-python/src/function_servicer.py
    # ... 
    # def Init(self, request, context):
    #     ...
    #     logger.info("init() Initialize MatrixOperation obj")
    #     self.top = TensorOperations()

    #     logger.info("init() Initialize Matrices") 
    #     self.top.initialize_tensors(2000,2000, 3)

    #     logger.info("init() finished!") 
    # ...                                                                               
    def initialize_tensors(self, depth, sizea, sizeb):                                 
        # Check if CUDA is available                                                   
        if not torch.cuda.is_available():                                              
            return                                                                     
                                                                                       
        # Ensure that the tensors are created on the GPU                               
        self.tensor_a = torch.randn(depth, sizea, sizeb, device='cuda')                
        self.tensor_b = torch.randn(depth, sizea, sizeb, device='cuda')                

    # Modify https://github.com/edgeless-project/runtime-python/src/function_servicer.py
    # ...
    # def Cast(self, request, context):
    #   ...
    #   logger.info("cast() Start multiplication")
    #   self.top.multiply_tensors(3)
    # ...                                                                                
    def multiply_tensors(self, repeat_counter):                                        
        # Check if CUDA is available                                                   
        if not torch.cuda.is_available():                                              
            return                                                                     
                                                                                       
        if self.tensor_a is None or self.tensor_b is None:                             
            return                                                                     
                                                                                       
        result = None                                                                  
                                                                                       
        # Perform multiplication in a loop to ensure it takes enough time              
        start_time = time.time()                                                       
        for _ in range(repeat_counter):                                                
            result = torch.matmul(self.tensor_a, self.tensor_b)                        
        end_time = time.time()                                                         
                                                                                       
        # Memory deallocation                                                         
        del self.tensor_a, self.tensor_b, result                                       
        gc.collect()                                                                   
                                                                                       
        return end_time                                                                
