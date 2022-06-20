from time import sleep
from glob import glob
import cv2
import numpy as np
import requests
import json

import AnnotationParser
from AnnotationParser import AnnotationParser
import ImageServer
from ImageServer import ImageServer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class StreamImage:
    def __init__(self,
                 imageDir,
                 imageProcessingEndpoint="",
                 imageProcessingParams="",
                 showImage = False,
                 verbose=True,
                 resizeWidth = 640,
                 resizeHeight = 640,
                 annotate=False,
                 sendToHubCallback = None):
        
        self.imageDir = imageDir
        self.showImage = showImage
        self.verbose = verbose
        self.resizeWidth = resizeWidth
        self.resizeHeight = resizeHeight
        self.imageProcessingEndpoint = imageProcessingEndpoint
        self.annotate = annotate
        self.sendToHubCallback = sendToHubCallback
        if imageProcessingParams == "":
            self.imageProcessingParams = "" 
        else:
            self.imageProcessingParams = json.loads(imageProcessingParams)
        
        if self.verbose:
            print("Initialising the camera capture with the following parameters: ")
            print("   - Image dir: " + self.imageDir)
            print("   - Show image: " + str(self.showImage))
            print("   - Image processing endpoint: " + self.imageProcessingEndpoint)
            print("   - Image processing params: " + json.dumps(self.imageProcessingParams))
            print("   - Resize width: " + str(self.resizeWidth))
            print("   - Resize height: " + str(self.resizeHeight))
            print("   - Annotate: " + str(self.annotate))
            print("   - Send processing results to hub: " + str(self.sendToHubCallback is not None))
            print()
        
        self.displayImage = None 
        
        if self.showImage:
            self.imageServer = ImageServer(5012, self)
            self.imageServer.start()
        
    def __annotate(self, frame, response):
        try:
            ap = AnnotationParser()
            listRectangles, listColors = ap.getAnnotations(response)
            for rectangle, color in zip(listRectangles, listColors):
                cv2.rectangle(frame, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), tuple(color), 3)
        except Exception as e:
            print(e)

    def __sendImageForProcessing(self, frame):
        headers = {'Content-Type': 'application/octet-stream'}
        try:
            response = requests.post(self.imageProcessingEndpoint, headers = headers, params = self.imageProcessingParams, data = frame)
        except Exception as e:
            print('__sendImageForProcessing Excpetion -' + str(e))
            return "[]"

        if self.verbose:
            try:
                print("Response from external processing service: (" + str(response.status_code) + ") " + \
                    json.dumps(response.json()["count"]))
            except Exception:
                print("Response from external processing service (status code): " + str(response.status_code))
        return json.dumps(response.json(), cls=NumpyEncoder)
    
    def get_display_image(self):
        return self.displayImage
    
    def start(self):
        images = glob(self.imageDir+'/*.*')
        for image in images:
            image = cv2.imread(image)

            #Pre-process locally
            if (self.resizeWidth != 0 or self.resizeHeight != 0):
                preprocessedImage = cv2.resize(image, (self.resizeWidth, self.resizeHeight))

            #Process externally
            if self.imageProcessingEndpoint != "":
                encodedImage = cv2.imencode(".jpg", preprocessedImage)[1].tostring()

                #Send over HTTP for processing
                response = self.__sendImageForProcessing(encodedImage)
                #forwarding outcome of external processing to the EdgeHub

                if response != "[]" and self.sendToHubCallback is not None:
                    self.sendToHubCallback(response)

            #Display frames
            if self.showImage:
                try:
                    if self.annotate and response != "[]":
                        predictions = json.loads(response)["predictions"]
                        self.__annotate(preprocessedImage, predictions)
                    self.displayImage = cv2.imencode('.jpg', preprocessedImage)[1].tobytes()
                except Exception as e:
                    print("Could not display the video to a web browser.") 
                    print('Excpetion -' + str(e))
            sleep(20)
    def __exit__(self, exception_type, exception_value, traceback):
        if self.showImage:
            self.imageServer.close()
                    
                    
if __name__ == "__main__":
    imageDir = "data" 
    sendImage = StreamImage(imageDir, showImage=True, annotate=True)
    sendImage.start()
        
        