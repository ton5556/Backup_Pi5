from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

while True:
    kit.servo[0].angle = 30
    #kit.servo[1].angle = 30
    time.sleep(1)

    kit.servo[0].angle = 90
    #kit.servo[1].angle = 90
    time.sleep(1)

    kit.servo[0].angle = 150
    #kit.servo[1].angle = 150
    time.sleep(1)