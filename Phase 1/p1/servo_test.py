from board import SCL, SDA
import busio

from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

import time

i2c = busio.I2C(SCL, SDA)

pca = PCA9685(i2c)
pca.frequency = 50

pan = servo.Servo(pca.channels[0])
tilt = servo.Servo(pca.channels[1])

while True:

    for angle in [0,45,90,135,180]:
        pan.angle = angle
        tilt.angle = angle

        print("Angle:", angle)

        time.sleep(1)