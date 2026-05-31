from picamera2 import Picamera2
import cv2
import time

from board import SCL, SDA
import busio

from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# =====================================
# PID CLASS
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

        derivative = 0

        if dt > 0:
            derivative = (error - self.previous_error) / dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.previous_error = error

        return output

# =====================================
# PCA9685
# =====================================

i2c = busio.I2C(SCL, SDA)

pca = PCA9685(i2c)
pca.frequency = 50

pan = servo.Servo(pca.channels[0])
tilt = servo.Servo(pca.channels[1])

# =====================================
# INITIAL POSITION
# =====================================

pan_angle = 90
tilt_angle = 90

pan.angle = pan_angle
tilt.angle = tilt_angle

# =====================================
# PID PARAMETERS
# =====================================

pid_pan = PID(
    kp=0.08,
    ki=0,
    kd=0.01
)

pid_tilt = PID(
    kp=0.08,
    ki=0,
    kd=0.01
)

# =====================================
# TARGET
# =====================================

target_x = 320
target_y = 240

# =====================================
# CAMERA
# =====================================

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)

picam2.configure(config)
picam2.start()

# =====================================
# MOUSE
# =====================================

def mouse_callback(event, x, y, flags, param):

    global target_x
    global target_y

    if event == cv2.EVENT_LBUTTONDOWN:

        target_x = x
        target_y = y

        print("Target:", target_x, target_y)

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_callback)

# =====================================
# LOOP
# =====================================

prev_time = time.time()

while True:

    frame = picam2.capture_array()

    now = time.time()
    dt = now - prev_time
    prev_time = now

    center_x = 320
    center_y = 240

    error_x = target_x - center_x
    error_y = target_y - center_y

    output_pan = pid_pan.update(error_x, dt)
    output_tilt = pid_tilt.update(error_y, dt)

    # -------------------------
    # PAN
    # -------------------------

    pan_angle -= output_pan

    pan_angle = max(
        0,
        min(
            180,
            pan_angle
        )
    )

    # -------------------------
    # TILT
    # -------------------------

    tilt_angle += output_tilt

    tilt_angle = max(
        0,
        min(
            180,
            tilt_angle
        )
    )

    pan.angle = pan_angle
    tilt.angle = tilt_angle

    # -------------------------
    # DRAW
    # -------------------------

    cv2.circle(
        frame,
        (center_x, center_y),
        6,
        (0,255,0),
        -1
    )

    cv2.circle(
        frame,
        (target_x, target_y),
        6,
        (0,0,255),
        -1
    )

    cv2.line(
        frame,
        (center_x, center_y),
        (target_x, target_y),
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"PAN {pan_angle:.1f}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"TILT {tilt_angle:.1f}",
        (10,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()

pca.deinit()