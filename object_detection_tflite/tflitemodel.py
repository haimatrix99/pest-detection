import logging
import tensorflow as tf
import numpy as np
from nms import non_max_suppression
from utils import plot_one_box, Colors, get_image_tensor
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TFLiteModel")

class TFLiteModel:
    def __init__(self, model_file, names_file, conf_thresh=0.25, iou_thresh=0.45,filter_classes=None, agnostic_nms=False, max_det=1000):
        self.model_file = model_file
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.agnostic_nms = agnostic_nms
        self.filter_classes = filter_classes
        self.max_det = max_det
        
        self.interpreter = None
        self.classes_name = [c.rstrip("\n") for c in open(names_file, 'r').readlines()]
        self.colors = Colors()
        self.inference = None
        
        self.make_interpreter()
        self.get_input_size()
        self.get_input_type()
    
    def make_interpreter(self):
        self.interpreter = tf.lite.Interpreter(self.model_file)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    def get_input_size(self):
        if self.interpreter is not None:
            self.input_size = self.input_details[0]['shape'][1], self.input_details[0]['shape'][1]
            return self.input_size
        else:
            logger.warn("Interpreter is not yet loaded")
            
    def get_input_type(self):
        if self.interpreter is not None:
            self.input_type = self.input_details[0]['dtype']
            return self.input_type
        else:
            logger.warn("Interpreter is not yet loaded")    
            
    def forward(self, x):
        if x.shape[0] == 3:
            x = x.transpose((1,2,0))
        
        x = x[np.newaxis].astype(np.float32)
        
        if self.input_type == np.uint8:
            self.input_zero = self.input_details[0]['quantization'][1]
            self.input_scale = self.input_details[0]['quantization'][0]
            self.output_zero = self.output_details[0]['quantization'][1]
            self.output_scale = self.output_details[0]['quantization'][0]
            
            if self.input_scale < 1e-9:
                self.input_scale = 1.0
            
            if self.output_scale < 1e-9:
                self.output_scale = 1.0
                
            x = (x/self.input_scale) + self.input_zero
            x = x.astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        if self.input_type == np.uint8:
            y = (y.astype(np.uint8) - self.output_zero) * self.output_scale

        y = non_max_suppression(y, self.conf_thresh, self.iou_thresh, self.filter_classes, self.agnostic_nms, self.max_det)
        
        return y
    
    def get_scaled_coords(self, xyxy, output_image, pad):
        """
        Converts raw prediction bounding box to orginal
        image coordinates.
        
        Args:
          xyxy: array of boxes
          output_image: np array
          pad: padding due to image resizing (pad_w, pad_h)
        """
        pad_w, pad_h = pad
        in_h, in_w = self.input_size
        out_h, out_w, _ = output_image.shape
                
        ratio_w = out_w/(in_w - pad_w)
        ratio_h = out_h/(in_h - pad_h) 
        
        out = []
        for coord in xyxy:

            x1, y1, x2, y2 = coord
                        
            x1 *= in_w*ratio_w
            x2 *= in_w*ratio_w
            y1 *= in_h*ratio_h
            y2 *= in_h*ratio_h
            
            x1 = max(0, x1)
            x2 = min(out_w, x2)
            
            y1 = max(0, y1)
            y2 = min(out_h, y2)
            
            out.append((x1, y1, x2, y2))
        
        return np.array(out).astype(int)
    
    def process_predictions(self, det, output_image, pad, hide_labels, hide_conf):
        """
        Process predictions and optionally output an image with annotations
        """
        if len(det):
            # Rescale boxes from img_size to im0 size
            # x1, y1, x2, y2=
            det[:, :4] = self.get_scaled_coords(det[:,:4], output_image, pad)
            
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (self.classes_name[c] if hide_conf else f'{self.classes_name[c]} {conf:.2f}')
                output_image = plot_one_box(xyxy, output_image, label=label, color=self.colors(c, True))
        return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
             
    def predict(self, image, hide_labels, hide_conf):
        full_image, net_image, pad = get_image_tensor(image, self.input_size[0])
        pred = self.forward(net_image)     
        det = self.process_predictions(pred[0], full_image, pad, hide_labels, hide_conf)
        return det       
        









