import cv2
import numpy as np
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

def nothing(x): pass

cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("H low",  "HSV Tuner", 0,   180, nothing)
cv2.createTrackbar("H high", "HSV Tuner", 10,  180, nothing)
cv2.createTrackbar("S low",  "HSV Tuner", 70,  255, nothing)
cv2.createTrackbar("V low",  "HSV Tuner", 50,  255, nothing)

while True:
    frame = picam2.capture_array()
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hl = cv2.getTrackbarPos("H low",  "HSV Tuner")
    hh = cv2.getTrackbarPos("H high", "HSV Tuner")
    sl = cv2.getTrackbarPos("S low",  "HSV Tuner")
    vl = cv2.getTrackbarPos("V low",  "HSV Tuner")

    mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, 255, 255]))

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask",  mask)

    if cv2.waitKey(1) == 27:
        break
    
    print(f"Lower: [{hl}, {sl}, {vl}]  Upper: [{hh}, 255, 255]")

cv2.destroyAllWindows()