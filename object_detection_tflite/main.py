from imutils.video import FileVideoStream
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='assets/best-fp32.tflite')

cap = FileVideoStream("data/VID-01.mp4").start()

while True:
    frame = cap.read()
    
    frame_resized = cv2.resize(frame, (640,640))
    frame_pred = model(frame_resized, size=640)
    frame_pred.render()
    cv2.imshow("Video", frame_resized)
    if cv2.waitKey(1) == ord("q"):
        break
    
cv2.destroyAllWindows()
cap.stop()



