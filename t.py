# Import required libraries
import cv2
import time
import threading
from adafruit_servokit import ServoKit
from sshkeyboard import listen_keyboard, stop_listening

# Initialize the ServoKit
kit = ServoKit(channels=16)

# Set default angles for pan and tilt
pan_angle = 90
tilt_angle = 90

kit.servo[0].angle = pan_angle
kit.servo[1].angle = tilt_angle

# Functions for detecting key presses
release_a = False
release_d = False
release_w = False
release_s = False
loop = True

def press(key):
    global loop, release_a, release_d, release_w, release_s

    if key == 'q':
        loop = False

    elif key == "w":
        release_w = False

    elif key == "s":
        release_s = False

    elif key == "a":
        release_a = False

    elif key == "d":
        release_d = False

def release(key):
    global release_a, release_d, release_w, release_s

    if key == "w":
        release_w = True

    elif key == "s":
        release_s = True

    elif key == "a":
        release_a = True

    elif key == "d":
        release_d = True

def input_keyboard():
    listen_keyboard(
        on_press=press,
        on_release=release,
        delay_second_char = 0.001
    )

keyboard_thread = threading.Thread(target=input_keyboard)
keyboard_thread.start()

# Connect to RTSP stream
cap = cv2.VideoCapture("rtsp://localhost:8554/cam")

# Read the first frame to determine resolution
ret, img = cap.read()
if not ret or img is None:
    raise RuntimeError("Unable to read initial frame from RTSP stream")

# Determine the target resolution based on aspect ratio
if img.shape[1]/img.shape[0] > 1.55:
    res = (256,144)
else:
    res = (216,162)

# Calculate positions and proportion values for tracking
XC = res[0]/2
XR = XC*33/32
XL = XC*31/32
XP = res[0]/5

YC = res[1]*7/16
YT = YC*31/32
YB = YC*33/32
YP = res[1]/5

# Initialize Haar cascade for face detection
cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Start processing video stream
while loop:
    # Read frame from video stream
    ret, img = cap.read()
    if not ret or img is None:
        continue

    # Handle manual control input first
    if not release_w and tilt_angle > 15:
        tilt_angle -= 1
    elif not release_s and tilt_angle < 180:
        tilt_angle += 1

    if not release_a and pan_angle > 0:
        pan_angle -= 1
    elif not release_d and pan_angle < 180:
        pan_angle += 1

    kit.servo[0].angle = pan_angle
    kit.servo[1].angle = tilt_angle

    # Resize frame to target resolution
    resized = cv2.resize(img, res, interpolation=cv2.INTER_AREA)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = cascade.detectMultiScale(gray)

    fw = 0
    fh = 0
    # Initialize face coordinates
    for (x,y,w,h) in faces:
        fx = x
        fy = y
        fw = w
        fh = h

    # Adjust pan and tilt angles based on detected face position
    if fw*fh > 0:
        center = ((fw/2)+fx, (fh/2)+fy)

        # Pan adjustment
        if center[0] < XL or center[0] > XR:
            pan_angle = round(pan_angle - (center[0] - XC) / XP)

        # Tilt adjustment
        if center[1] < YT or center[1] > YB:
            tilt_angle = round(tilt_angle + (center[1] - YC) / YP)

        # Ensure angles are within valid range (0-180)
        if pan_angle > 180:
            pan_angle = 180
        elif pan_angle < 0:
            pan_angle = 0
        if tilt_angle > 180:
            tilt_angle = 180
        elif tilt_angle < 0:
            tilt_angle = 0

        # Update servo angles
        kit.servo[0].angle = pan_angle
        kit.servo[1].angle = tilt_angle

# Release resources
cap.release()
stop_listening()
keyboard_thread.join()
print("Exiting")