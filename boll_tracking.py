#!/usr/bin/env python3

from collections import deque
from picamera2 import Picamera2
from libcamera import controls
import cv2
import numpy as np
import time

# ==========================================
# Green Ball HSV Range
# ==========================================

GREEN_LOWER = (29, 86, 6)
GREEN_UPPER = (64, 255, 255)

BUFFER_SIZE = 64
pts = deque(maxlen=BUFFER_SIZE)

# ==========================================
# Picamera2 Setup
# ==========================================

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (2304, 1296),
        "format": "RGB888"
    },
    buffer_count=4
)

picam2.configure(config)

picam2.start()

time.sleep(2)

# ==========================================
# Camera Quality Settings
# ==========================================

picam2.set_controls({

    # Auto Focus
    "AfMode": controls.AfModeEnum.Continuous,

    # Auto Exposure
    "AeEnable": True,

    # Image Enhancement
    "Sharpness": 2.0,
    "Contrast": 1.3,
    "Brightness": 0.05,
    "Saturation": 1.1,

    # Fast shutter for moving objects
    "ExposureTime": 3000
})

# ==========================================
# FPS Counter
# ==========================================

fps_start = time.time()
fps_frames = 0

# ==========================================
# Main Loop
# ==========================================

while True:

    frame = picam2.capture_array()

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    # Resize for processing
    frame = cv2.resize(
        frame,
        (960, 540),
        interpolation=cv2.INTER_AREA
    )

    # ======================================
    # Blur + HSV
    # ======================================

    blurred = cv2.GaussianBlur(
        frame,
        (11, 11),
        0
    )

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # ======================================
    # Green Mask
    # ======================================

    mask = cv2.inRange(
        hsv,
        GREEN_LOWER,
        GREEN_UPPER
    )

    mask = cv2.erode(
        mask,
        None,
        iterations=2
    )

    mask = cv2.dilate(
        mask,
        None,
        iterations=2
    )

    # ======================================
    # Find Contours
    # ======================================

    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    center = None

    if len(contours) > 0:

        c = max(
            contours,
            key=cv2.contourArea
        )

        ((x, y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)

        if M["m00"] > 0:

            center = (
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"])
            )

            if radius > 10:

                # Circle
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    int(radius),
                    (0, 255, 255),
                    2
                )

                # Center
                cv2.circle(
                    frame,
                    center,
                    5,
                    (0, 0, 255),
                    -1
                )

                # Coordinates
                cv2.putText(
                    frame,
                    f"X:{center[0]} Y:{center[1]}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Radius
                cv2.putText(
                    frame,
                    f"Radius:{int(radius)}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # ======================================
    # Track Trail
    # ======================================

    pts.appendleft(center)

    for i in range(1, len(pts)):

        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(
            np.sqrt(
                BUFFER_SIZE / float(i + 1)
            ) * 2.5
        )

        cv2.line(
            frame,
            pts[i - 1],
            pts[i],
            (0, 0, 255),
            thickness
        )

    # ======================================
    # Draw Crosshair
    # ======================================

    h, w = frame.shape[:2]

    center_x = w // 2
    center_y = h // 2

    cv2.line(
        frame,
        (center_x - 20, center_y),
        (center_x + 20, center_y),
        (255, 0, 0),
        2
    )

    cv2.line(
        frame,
        (center_x, center_y - 20),
        (center_x, center_y + 20),
        (255, 0, 0),
        2
    )

    # ======================================
    # FPS
    # ======================================

    fps_frames += 1

    elapsed = time.time() - fps_start

    fps = fps_frames / elapsed

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # ======================================
    # Display
    # ======================================

    cv2.imshow(
        "Green Ball Tracking",
        frame
    )

    cv2.imshow(
        "Mask",
        mask
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# ==========================================
# Cleanup
# ==========================================

cv2.destroyAllWindows()
picam2.stop()