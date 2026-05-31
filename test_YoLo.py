#!/usr/bin/env python3

import cv2
import time
import numpy as np

from ultralytics import YOLO
from picamera2 import Picamera2
from adafruit_servokit import ServoKit

# ==================================================
# CAMERA
# ==================================================

FRAME_W = 640
FRAME_H = 480

CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

# ==================================================
# SERVO
# ==================================================

PAN_CHANNEL = 0
TILT_CHANNEL = 1

PAN_MIN = 0
PAN_MAX = 180

TILT_MIN = 30
TILT_MAX = 150

PAN_START = 90
TILT_START = 90

# ==================================================
# PID
# ==================================================

class PID:

    def __init__(self, kp, ki, kd):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()

    def reset(self):

        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()

    def compute(self, error):

        now = time.time()

        dt = now - self.prev_time

        if dt <= 0:
            dt = 0.001

        self.integral += error * dt

        derivative = (error - self.prev_error) / dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.prev_error = error
        self.prev_time = now

        return output


# ==================================================
# INIT SERVO
# ==================================================

print("Init PCA9685...")

kit = ServoKit(channels=16)

pan_angle = float(PAN_START)
tilt_angle = float(TILT_START)

kit.servo[PAN_CHANNEL].angle = pan_angle
kit.servo[TILT_CHANNEL].angle = tilt_angle

time.sleep(1)

# ==================================================
# PID TUNING
# ==================================================

pid_pan = PID(
    kp=0.04,
    ki=0.0,
    kd=0.002
)

pid_tilt = PID(
    kp=0.04,
    ki=0.0,
    kd=0.002
)

DEADZONE = 15

# ==================================================
# CAMERA
# ==================================================

print("Starting Camera...")

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (FRAME_W, FRAME_H),
        "format": "BGR888"
    }
)

picam2.configure(config)

picam2.start()

time.sleep(2)

# ==================================================
# YOLO
# ==================================================

print("Loading YOLO...")

model = YOLO("yolov8n.pt")

print("YOLO Ready")

# ==================================================
# SETTINGS
# ==================================================

TARGET_CLASS = "person"

frame_count = 0

last_results = None

prev_time = time.time()

# ==================================================
# LOOP
# ==================================================

try:

    while True:

        frame = picam2.capture_array()

        frame_count += 1

        # ----------------------------
        # YOLO every 3 frames
        # ----------------------------

        if frame_count % 3 == 0:

            last_results = model(
                frame,
                imgsz=320,
                verbose=False
            )

        results = last_results

        target = None
        largest_area = 0

        # ----------------------------
        # FIND TARGET
        # ----------------------------

        if results is not None:

            for result in results:

                for box in result.boxes:

                    conf = float(box.conf[0])

                    if conf < 0.5:
                        continue

                    cls = int(box.cls[0])

                    label = model.names[cls]

                    if label != TARGET_CLASS:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    area = (x2 - x1) * (y2 - y1)

                    if area > largest_area:

                        largest_area = area

                        target = (
                            x1,
                            y1,
                            x2,
                            y2,
                            label,
                            conf
                        )

        # ----------------------------
        # TRACK TARGET
        # ----------------------------

        if target is not None:

            x1, y1, x2, y2, label, conf = target

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            err_x = cx - CENTER_X
            err_y = cy - CENTER_Y

            cv2.line(
                frame,
                (CENTER_X, CENTER_Y),
                (cx, cy),
                (0, 255, 255),
                2
            )

            # PAN

            if abs(err_x) > DEADZONE:

                pan_output = pid_pan.compute(err_x)

                pan_angle -= pan_output

            else:

                pid_pan.reset()

            # TILT

            if abs(err_y) > DEADZONE:

                tilt_output = pid_tilt.compute(err_y)

                tilt_angle += tilt_output

            else:

                pid_tilt.reset()

            pan_angle = max(
                PAN_MIN,
                min(PAN_MAX, pan_angle)
            )

            tilt_angle = max(
                TILT_MIN,
                min(TILT_MAX, tilt_angle)
            )

            kit.servo[PAN_CHANNEL].angle = pan_angle
            kit.servo[TILT_CHANNEL].angle = tilt_angle

        # ----------------------------
        # CROSSHAIR
        # ----------------------------

        cv2.line(
            frame,
            (CENTER_X - 20, CENTER_Y),
            (CENTER_X + 20, CENTER_Y),
            (255,255,255),
            1
        )

        cv2.line(
            frame,
            (CENTER_X, CENTER_Y - 20),
            (CENTER_X, CENTER_Y + 20),
            (255,255,255),
            1
        )

        cv2.circle(
            frame,
            (CENTER_X, CENTER_Y),
            DEADZONE,
            (200,200,200),
            1
        )

        # ----------------------------
        # FPS
        # ----------------------------

        now = time.time()

        fps = 1 / (now - prev_time)

        prev_time = now

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Pan:{int(pan_angle)}",
            (10,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Tilt:{int(tilt_angle)}",
            (10,90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.imshow(
            "YOLO PID Tracking",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):

            pan_angle = PAN_START
            tilt_angle = TILT_START

            kit.servo[PAN_CHANNEL].angle = pan_angle
            kit.servo[TILT_CHANNEL].angle = tilt_angle

except KeyboardInterrupt:
    pass

finally:

    kit.servo[PAN_CHANNEL].angle = PAN_START
    kit.servo[TILT_CHANNEL].angle = TILT_START

    picam2.stop()

    cv2.destroyAllWindows()

    print("Done")