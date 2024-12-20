from ultralytics import YOLO

# Modify https://github.com/edgeless-project/runtime-python/src/function_servicer.py
# from yolo_inference import *

class YoloOperations:
    def __init__(self):
        self.model = None
        self.trt_model = None
        # Modify filenames and paths if needed
        self.image_source = "/app/src/yolo_test_road.jpg"
        self.video_source = "/app/src/Traffic_IP_Camera_video.mp4"

    # Modify https://github.com/edgeless-project/runtime-python/src/function_servicer.py
    # ... 
    # def Init(self, request, context):
    #     ...
    #     logger.info("init() Initialize YoloOperations obj")
    #     self.top = YoloOperations()
    #     logger.info("init() Loading PyTorch model") 
    #     self.top.load_model()
    #     logger.info("init() finished!") 
    # ...   
    def load_model(self):
        """Load a pre-trained PyTorch model"""
        self.model = YOLO("yolo11n.pt")
    
    # Model conversion uses the GPU in a time consuming process, better not to use it in Init. 
    def convert_model(self):
        """Convert the loaded model to TensorRT format, 'yolo*.engine' is created"""
        self.model.export(format="engine")
        self.trt_model = YOLO("yolo11n.engine")

    def image_inference(self, model: str):
        if model == "pytorch":
            results = self.model(self.image_source)
        elif model == "tensorrt":
            results = self.trt_model(self.image_source)
        else:
            raise ValueError(
                f"Invalid model: '{model}'. Please use 'pytorch' or 'tensorrt'."
            )
    # Modify https://github.com/edgeless-project/runtime-python/src/function_servicer.py
    # ...
    # def Cast(self, request, context):
    #   ...
    #   logger.info("cast() Starting video inference with YOLO")
    #   self.top.video_inference('pytorch')
    # ...
    def video_inference(self, model: str):
        if model == "pytorch":
            self.model.predict(self.video_source, save=True, show=False, conf=0.5)
        elif model == "tensorrt":
            self.trt_model.predict(self.video_source, save=True, show=False, conf=0.5)
        else:
            raise ValueError(
                f"Invalid model: '{model}'. Please use 'pytorch' or 'tensorrt'."
            )