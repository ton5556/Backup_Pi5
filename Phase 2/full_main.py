from picamera2 import Picamera2
import cv2
import numpy as np
import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# =====================================
# PID CLASS (เหมือนเดิม)
# =====================================
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

# =====================================
# SERVO SETUP (เหมือนเดิม)
# =====================================
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50

pan  = servo.Servo(pca.channels[0])
tilt = servo.Servo(pca.channels[1])

pan_angle  = 90
tilt_angle = 90
pan.angle  = pan_angle
tilt.angle = tilt_angle

# =====================================
# PID
# =====================================
pid_pan  = PID(kp=0.08, ki=0.0, kd=0.03)
pid_tilt = PID(kp=0.08, ki=0.0, kd=0.03)

# =====================================
# HSV RANGE สีแดง — ปรับตรงนี้!
# =====================================
RED_LOWER1 = np.array([0,   70,  50])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([170, 70,  50])
RED_UPPER2 = np.array([180, 255, 255])

MIN_AREA = 500   # กรอง noise เล็กๆ ออก

# =====================================
# CAMERA
# =====================================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

prev_time = time.time()

while True:
    frame = picam2.capture_array()

    now = time.time()
    dt  = now - prev_time
    prev_time = now

    # --- HSV + MASK ---
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # optional: ลด noise ด้วย morphology
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.erode(mask,  kernel, iterations=1)
    mask   = cv2.dilate(mask, kernel, iterations=2)

    # --- CONTOUR ---
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cx, cy = 320, 240   # default = กลางจอ (servo ไม่ขยับ)
    found  = False

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)

        if area > MIN_AREA:
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx    = int(M["m10"] / M["m00"])
                cy    = int(M["m01"] / M["m00"])
                found = True

                # วาดวงกลมและ bounding circle
                ((bx, by), radius) = cv2.minEnclosingCircle(largest)
                cv2.circle(frame, (int(bx), int(by)), int(radius), (0, 255, 0), 2)

    # --- PID (เหมือน Phase 1 ทุกอย่าง!) ---
    if found:
        error_x = cx - 320
        error_y = cy - 240

        pan_angle  -= pid_pan.update(error_x, dt)
        tilt_angle += pid_tilt.update(error_y, dt)

        pan_angle  = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))

        pan.angle  = pan_angle
        tilt.angle = tilt_angle

    # --- DRAW ---
    cv2.circle(frame, (320, 240), 6, (0, 255, 0), -1)   # center
    cv2.circle(frame, (cx, cy),   6, (0, 0, 255), -1)   # target

    status = "TRACKING" if found else "SEARCHING..."
    color  = (0, 255, 0) if found else (0, 0, 255)
    cv2.putText(frame, status,           (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"PAN  {pan_angle:.1f}",  (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"TILT {tilt_angle:.1f}", (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"AREA {area:.0f}" if found else "AREA -", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Color Tracking", frame)
    cv2.imshow("Mask", mask)   # ดู mask แยกต่างหาก ช่วย debug มาก!

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
pca.deinit()