import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from imutils.video import FPS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tflitemodel import TFLiteModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser("EdgeTPU test runner")
    parser.add_argument("--model", "-m", type=str, default="assets/best-fp32-v2.tflite", help="weights file")
    parser.add_argument("--names", "-n", type=str, default="assets/classes.txt", help="classes name file")
    parser.add_argument("--conf-thresh", type=float, default=0.5, help="model confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--hide-labels", action='store_true', help="Hide labels")
    parser.add_argument("--hide-conf", action='store_true', help="Hide confidence")    
    parser.add_argument("--image", "-i", type=str, help="Image file to run detection on")
    parser.add_argument("--stream", "-s", type=str, help="Stream file or camera to run detection on")

    
    args = parser.parse_args()
    
    model = TFLiteModel(args.model, args.names, args.conf_thresh, args.iou_thresh)
    
    input_size = model.get_input_size()

    x = (255*np.random.random((3,*input_size)))
    model.forward(x)
    
    if args.image is not None:
        image = np.array(Image.open(args.image))
        image_resized = cv2.resize(image, (640,640))
        pred = model.predict(image_resized, args.hide_labels, args.hide_conf)
        cv2.imshow("Predicted", pred)
        cv2.waitKey(0)
        
    elif args.stream is not None:
        cap = cv2.VideoCapture(args.stream)
        fps = FPS().start()
        while True:
            _, frame = cap.read()
            frame_flip = cv2.flip(frame, 0)
            image_rgb = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (640,640))
            pred = model.predict(image_resized, args.hide_labels, args.hide_conf)
            cv2.imshow("Predicted", pred)
            if cv2.waitKey(1) == ord("q"):
                break
            fps.update()
        fps.stop()
        print("fps elapsed time:", fps.elapsed())
        print("fps inference:", fps.fps())
        cap.release()
        cv2.destroyAllWindows()