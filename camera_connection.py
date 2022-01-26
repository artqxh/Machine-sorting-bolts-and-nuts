import cv2
import time

cap = cv2.VideoCapture(-1)
ret, frame = cap.read()

def takePicture():
    (grabbed, frame) = cap.read()
    showimg = frame
    cv2.imshow('img1', showimg)
    cv2.waitKey(1)
    time.sleep(0.3)
    image = 'result.png'
    cv2.imwrite(image, frame)
    cap.release()
    return image

print(takePicture())