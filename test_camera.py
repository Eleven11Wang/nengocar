import Camera 
import cv2
import time
import camera_test 

cv2.namedWindow("frame")
MyCamera = camera_test.USBCamera((480,360))
while True:
    frame = MyCamera.getframe()
    if frame is not None:
        cv2.imshow("frame",frame)
        cv2.imwrite("test_frame.jpg",frame)
        break

MyCamera.shutdown()
