from picamera2 import Picamera2
from adafruit_servokit import ServoKit
import cv2
import numpy as np
import time

# =========================
# PID Controller
# =========================

class PID:
    """
    Tuning:
      1. Ki=0, Kd=0 ? ????? Kp ??????????????????
      2. ????? Kd ???? 0.1 ??????????
      3. ????? Ki ??????????? servo ??????????
    """
    def __init__(self, Kp, Ki, Kd, out_min, out_max, integral_limit=30.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.out_min, self.out_max = out_min, out_max
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = max(now - self._prev_time, 1e-6)

        self._integral = np.clip(
            self._integral + error * dt,
            -self.integral_limit, self.integral_limit
        )
        d = self.Kd * (error - self._prev_error) / dt
        output = np.clip(
            self.Kp * error + self.Ki * self._integral + d,
            self.out_min, self.out_max
        )
        self._prev_error = error
        self._prev_time = now
        return float(output)

# =========================
# Servo Setup
# =========================

kit = ServoKit(channels=16)
pan  = 90.0
tilt = 90.0
kit.servo[0].angle = pan
kit.servo[1].angle = tilt

# PID gains � ??????????
DEADZONE = 20   # px � ??? error ??????????? ???????

pid_pan  = PID(Kp=0.04, Ki=0.0, Kd=0.0015, out_min=-10, out_max=10)
pid_tilt = PID(Kp=0.04, Ki=0.0, Kd=0.0015, out_min=-10, out_max=10)

# =========================
# Camera Setup
# =========================

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480)}
))
picam2.start()

cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("H Low",  "HSV Tuner", 100, 179, lambda x: None)
cv2.createTrackbar("H High", "HSV Tuner", 130, 179, lambda x: None)
cv2.createTrackbar("S Low",  "HSV Tuner", 150, 255, lambda x: None)

# =========================
# Main Loop
# =========================

FRAME_CX, FRAME_CY = 320, 240

while True:
    frame = picam2.capture_array()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  120,  70])
    upper = np.array([10, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        cy = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        err_x = cx - FRAME_CX   # + = ???????????
        err_y = cy - FRAME_CY   # + = ????????????

        # Pan: ??????????? ? pan ?? (???????????)
        if abs(err_x) > DEADZONE:
            pan -= pid_pan.compute(err_x)
        else:
            pid_pan.reset()

        # Tilt: ???????????? ? tilt ????? (????????)
        if abs(err_y) > DEADZONE:
            tilt += pid_tilt.compute(err_y)
        else:
            pid_tilt.reset()

        pan  = float(np.clip(pan,  0, 180))
        tilt = float(np.clip(tilt, 0, 180))

        kit.servo[0].angle = pan
        kit.servo[1].angle = tilt

    else:
        # ?????????? � reset integrator ??????? windup
        pid_pan.reset()
        pid_tilt.reset()

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()