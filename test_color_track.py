from picamera2 import Picamera2
from adafruit_servokit import ServoKit
import cv2
import numpy as np

# =========================
# Servo Setup
# =========================
kit = ServoKit(channels=16)

pan = 90
tilt = 90

kit.servo[0].angle = pan
kit.servo[1].angle = tilt

# =========================
# Camera Setup
# =========================
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)

picam2.configure(config)
picam2.start()

cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("H Low",  "HSV Tuner", 100, 179, lambda x: None)
cv2.createTrackbar("H High", "HSV Tuner", 130, 179, lambda x: None)
cv2.createTrackbar("S Low",  "HSV Tuner", 150, 255, lambda x: None)

# =========================
# Main Loop
# =========================
while True:

    frame = picam2.capture_array()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color range
    lower = np.array([0, 120, 70])
    upper = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)

        cx = x + w // 2
        cy = y + h // 2

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

        # Frame center
        fx = 320
        fy = 240

        # Pan control
        if cx < fx - 30:
            pan += 1

        elif cx > fx + 30:
            pan -= 1

        # Tilt control
        if cy < fy - 30:
            tilt -= 1

        elif cy > fy + 30:
            tilt += 1

        # Limit angles
        pan = max(0, min(180, pan))
        tilt = max(0, min(180, tilt))

        # Move servos
        kit.servo[0].angle = pan
        kit.servo[1].angle = tilt

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()