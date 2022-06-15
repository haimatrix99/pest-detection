import logging
import tensorflow as tf
import numpy as np
from nms import non_max_suppression
from utils import plot_one_box, Colors, get_image_tensor, xyxy2xywh
import cv2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TFLiteModel")

class TFLiteModel:
    def __init__(self, model_file, deep_sort_model, config_deep_sort, names_file, conf_thresh=0.25, iou_thresh=0.45,filter_classes=None, agnostic_nms=False, max_det=1000):
        self.model_file = model_file
        self.deep_sort_model = deep_sort_model
        self.config_deep_sort = config_deep_sort
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
        self.make_deep_sort()
        self.get_input_size()
    
    def make_deep_sort(self):
        cfg = get_config()
        cfg.merge_from_file(self.config_deep_sort)
        self.deepsort = DeepSort(self.deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
        
    def make_interpreter(self):
        self.interpreter = tf.lite.Interpreter(self.model_file)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def get_input_size(self):
        if self.interpreter is not None:
            self.input_size = (self.input_details[0]['shape'][1], self.input_details[0]['shape'][1])
            return self.input_size
        else:
            logger.warn("Interpreter is not yet loaded")
            
    def forward(self, x):
        if x.shape[0] == 3:
            x = x.transpose((1,2,0))
        
        x = x[np.newaxis].astype(np.float32)
                    
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        
        y = self.interpreter.get_tensor(self.output_details[0]['index'])
    
        y = non_max_suppression(y, self.conf_thresh, self.iou_thresh, self.filter_classes, self.agnostic_nms, self.max_det)
        return y
    
    def get_scaled_coords(self, xyxy, output_image, pad):
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
        if len(det) > 0:
            det[:, :4] = self.get_scaled_coords(det[:,:4], output_image, pad)
            
            xywhs = xyxy2xywh(det[:, :4])
            confs = det[:, 4]
            clss = det[:, 5]
            
            outputs = self.deepsort.update(xywhs, confs, clss, output_image)
            if len(outputs) > 0:
                for _, (output, conf) in enumerate(zip(outputs, confs)):
                    if conf > 0.5:
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                        label = f'{id}'
                        output_image = plot_one_box(bboxes, output_image, label=label, color=self.colors(c, True))
        else:
            self.deepsort.increment_ages()
        return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
             
    def predict(self, image, hide_labels, hide_conf):
        full_image, net_image, pad = get_image_tensor(image, self.input_size[0])
        pred = self.forward(net_image)     
        dets = self.process_predictions(pred[0], full_image, pad, hide_labels, hide_conf)
        return dets    
        









