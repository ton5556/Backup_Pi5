from picamera2 import Picamera2
import cv2

from board import SCL,SDA
import busio

from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# =====================
# PCA9685
# =====================

i2c = busio.I2C(SCL,SDA)

pca = PCA9685(i2c)
pca.frequency = 50

pan = servo.Servo(pca.channels[0])
tilt = servo.Servo(pca.channels[1])

pan_angle = 90
tilt_angle = 90

pan.angle = pan_angle
tilt.angle = tilt_angle

# =====================
# CAMERA
# =====================

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size":(640,480)}
)

picam2.configure(config)
picam2.start()

# =====================
# MOUSE
# =====================

def mouse_callback(event,x,y,flags,param):

    global pan_angle
    global tilt_angle

    if event == cv2.EVENT_LBUTTONDOWN:

        print("Click:",x,y)

        pan_angle = 180 - int(x/640*180)

        tilt_angle = int(y/480*180)

        pan_angle = max(0,min(180,pan_angle))
        tilt_angle = max(0,min(180,tilt_angle))

        pan.angle = pan_angle
        tilt.angle = tilt_angle

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera",mouse_callback)


class PID:

    def __init__(self,kp,ki,kd):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0
        self.integral = 0

    def update(self,error):

        self.integral += error

        derivative = error - self.prev_error

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.prev_error = error

        return output

# =====================
# LOOP
# =====================

while True:

    frame = picam2.capture_array()

    cv2.circle(frame,(320,240),5,(0,255,0),-1)

    cv2.imshow("Camera",frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()