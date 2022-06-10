import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best-fp16.tflite')  # TFLite


