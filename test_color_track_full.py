#!/usr/bin/env python3
"""
=============================================================
  Raspberry Pi 5 + Camera Module 3
  2-Axis Color Tracking — Phase 2  (Real PID Control)
  Hardware: Pi 5 · Camera Module 3 · PCA9685 · SG90 x2
=============================================================
  PID replaces the simple threshold (+1/-1) from Phase 1.
  Servo movement is now proportional to error — smooth,
  fast when far, gentle when close, no oscillation.

  Controls:
    Q        — quit
    R        — reset servo to center (90°, 90°)
    S        — save HSV + PID config to config.json
    L        — load config from config.json
    SPACE    — pause / resume tracking
    P        — toggle PID debug overlay
=============================================================
"""

import cv2
import numpy as np
import json
import os
import time
from picamera2 import Picamera2
from adafruit_servokit import ServoKit

# =============================================================
# Hardware Config
# =============================================================

FRAME_W = 640
FRAME_H = 480
CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

PAN_CHANNEL  = 0
TILT_CHANNEL = 1

PAN_MIN  = 0
PAN_MAX  = 180
TILT_MIN = 30
TILT_MAX = 150

PAN_START  = 90
TILT_START = 90

MIN_CONTOUR_AREA = 800

CONFIG_FILE = "config.json"

# =============================================================
# PID Controller
# =============================================================

class PID:
    """
    Standard PID controller.

    output = Kp*error + Ki*integral + Kd*derivative

    - Kp (Proportional): ยิ่ง error มาก ยิ่งขยับเร็ว
    - Ki (Integral): แก้ offset สะสม (ส่วนใหญ่ไม่จำเป็นสำหรับ servo)
    - Kd (Derivative): เบรกเมื่อเข้าใกล้เป้า ป้องกัน overshoot

    Tuning guide:
      1. ตั้ง Ki=0, Kd=0  แล้วเพิ่ม Kp จนติดตามได้แต่ยังสั่น
      2. เพิ่ม Kd ทีละ 0.1 จนหยุดสั่น
      3. ถ้า servo ไม่ไปถึงเป้าเป๊ะ ค่อยเพิ่ม Ki นิดหน่อย
    """

    def __init__(self, Kp: float, Ki: float, Kd: float,
                 out_min: float, out_max: float,
                 integral_limit: float = 30.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral_limit = integral_limit  # anti-windup

        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def compute(self, error: float) -> float:
        now = time.time()
        dt  = now - self._prev_time
        if dt <= 0:
            dt = 1e-6

        # Proportional
        p = self.Kp * error

        # Integral with anti-windup clamp
        self._integral += error * dt
        self._integral  = max(-self.integral_limit,
                              min(self.integral_limit, self._integral))
        i = self.Ki * self._integral

        # Derivative (on measurement, not error — avoids derivative kick)
        d = self.Kd * (error - self._prev_error) / dt

        output = p + i + d
        output = max(self.out_min, min(self.out_max, output))

        self._prev_error = error
        self._prev_time  = now

        return output

    @property
    def components(self) -> tuple[float, float, float]:
        """Return last P, I, D values for debug display."""
        e  = self._prev_error
        return (self.Kp * e,
                self.Ki * self._integral,
                self.Kd * 0.0)   # D stored separately if needed


# =============================================================
# Default Config  (HSV + PID gains)
# =============================================================

DEFAULT_CONFIG = {
    # HSV — red object (phase 1 default was blue; change back to blue
    # by setting h_low=100, h_high=130)
    "h_low":   0,
    "h_high": 10,
    "s_low":  120,
    "s_high": 255,
    "v_low":   70,
    "v_high": 255,

    # PID gains — pan axis
    # output unit = degrees/frame, clamped to ±10
    "pan_kp":  0.04,
    "pan_ki":  0.0,
    "pan_kd":  0.001,

    # PID gains — tilt axis
    "tilt_kp":  0.04,
    "tilt_ki":  0.0,
    "tilt_kd":  0.001,

    # Dead-zone: error (px) inside which PID output is ignored
    "deadzone": 15,
}

# =============================================================
# Config helpers
# =============================================================

def save_config(cfg: dict, path: str = CONFIG_FILE):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Config saved → {path}")

def load_config(path: str = CONFIG_FILE) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            cfg = {**DEFAULT_CONFIG, **json.load(f)}
        print(f"[INFO] Config loaded ← {path}")
        return cfg
    print("[INFO] No config file found, using defaults")
    return DEFAULT_CONFIG.copy()

# =============================================================
# Trackbar windows
# =============================================================

HSV_WIN = "HSV Tuner"
PID_WIN = "PID Tuner"

def _tb(name, win, val, mx):
    cv2.createTrackbar(name, win, val, mx, lambda x: None)

def create_hsv_trackbars(cfg: dict):
    cv2.namedWindow(HSV_WIN)
    _tb("H Low",  HSV_WIN, cfg["h_low"],  179)
    _tb("H High", HSV_WIN, cfg["h_high"], 179)
    _tb("S Low",  HSV_WIN, cfg["s_low"],  255)
    _tb("S High", HSV_WIN, cfg["s_high"], 255)
    _tb("V Low",  HSV_WIN, cfg["v_low"],  255)
    _tb("V High", HSV_WIN, cfg["v_high"], 255)

def create_pid_trackbars(cfg: dict):
    """
    Trackbar stores integer 0-200; divide by 1000 for Kp/Kd,
    divide by 10000 for Ki (usually tiny).
    """
    cv2.namedWindow(PID_WIN)
    _tb("Pan  Kp x1000",  PID_WIN, int(cfg["pan_kp"]  * 1000), 200)
    _tb("Pan  Kd x1000",  PID_WIN, int(cfg["pan_kd"]  * 1000), 200)
    _tb("Pan  Ki x10000", PID_WIN, int(cfg["pan_ki"]  * 10000), 50)
    _tb("Tilt Kp x1000",  PID_WIN, int(cfg["tilt_kp"] * 1000), 200)
    _tb("Tilt Kd x1000",  PID_WIN, int(cfg["tilt_kd"] * 1000), 200)
    _tb("Tilt Ki x10000", PID_WIN, int(cfg["tilt_ki"] * 10000), 50)
    _tb("Deadzone px",    PID_WIN, cfg["deadzone"], 80)

def read_hsv_trackbars() -> dict:
    return {
        "h_low":  cv2.getTrackbarPos("H Low",  HSV_WIN),
        "h_high": cv2.getTrackbarPos("H High", HSV_WIN),
        "s_low":  cv2.getTrackbarPos("S Low",  HSV_WIN),
        "s_high": cv2.getTrackbarPos("S High", HSV_WIN),
        "v_low":  cv2.getTrackbarPos("V Low",  HSV_WIN),
        "v_high": cv2.getTrackbarPos("V High", HSV_WIN),
    }

def read_pid_trackbars() -> dict:
    return {
        "pan_kp":  cv2.getTrackbarPos("Pan  Kp x1000",  PID_WIN) / 1000,
        "pan_kd":  cv2.getTrackbarPos("Pan  Kd x1000",  PID_WIN) / 1000,
        "pan_ki":  cv2.getTrackbarPos("Pan  Ki x10000", PID_WIN) / 10000,
        "tilt_kp": cv2.getTrackbarPos("Tilt Kp x1000",  PID_WIN) / 1000,
        "tilt_kd": cv2.getTrackbarPos("Tilt Kd x1000",  PID_WIN) / 1000,
        "tilt_ki": cv2.getTrackbarPos("Tilt Ki x10000", PID_WIN) / 10000,
        "deadzone": cv2.getTrackbarPos("Deadzone px",   PID_WIN),
    }

def sync_pid_gains(pid_pan: PID, pid_tilt: PID, cfg: dict):
    """Push trackbar values into live PID objects."""
    pid_pan.Kp  = cfg["pan_kp"]
    pid_pan.Ki  = cfg["pan_ki"]
    pid_pan.Kd  = cfg["pan_kd"]
    pid_tilt.Kp = cfg["tilt_kp"]
    pid_tilt.Ki = cfg["tilt_ki"]
    pid_tilt.Kd = cfg["tilt_kd"]

# =============================================================
# Drawing helpers
# =============================================================

FONT        = cv2.FONT_HERSHEY_SIMPLEX
C_BOX       = (0, 255, 0)
C_DOT       = (0, 0, 255)
C_CROSS     = (180, 180, 180)
C_WHITE     = (255, 255, 255)
C_YELLOW    = (0, 220, 255)
C_CYAN      = (255, 220, 0)

def _text(frame, txt, x, y, scale=0.42, color=C_WHITE):
    cv2.putText(frame, txt, (x, y), FONT, scale, (0, 0, 0),   2, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), FONT, scale,      color,  1, cv2.LINE_AA)

def draw_crosshair(frame, deadzone: int):
    cx, cy = CENTER_X, CENTER_Y
    cv2.line(frame, (cx - 22, cy), (cx + 22, cy), C_CROSS, 1)
    cv2.line(frame, (cx, cy - 22), (cx, cy + 22), C_CROSS, 1)
    cv2.circle(frame, (cx, cy), deadzone, C_CROSS, 1)

def draw_target(frame, x, y, w, h, cx, cy):
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_BOX, 2)
    cv2.circle(frame, (cx, cy), 5, C_DOT, -1)
    cv2.line(frame, (CENTER_X, CENTER_Y), (cx, cy), C_YELLOW, 1)
    # error vector label
    ex = cx - CENTER_X
    ey = cy - CENTER_Y
    _text(frame, f"ex={ex:+d} ey={ey:+d}", cx + 8, cy - 8, color=C_YELLOW)

def draw_hud(frame, pan, tilt, hsv, pid_cfg, fps, paused, tracking, show_pid):
    lines = [
        f"Pan:{pan:6.2f}  Tilt:{tilt:6.2f}",
        f"HSV H[{hsv['h_low']}-{hsv['h_high']}] S[{hsv['s_low']}-{hsv['s_high']}] V[{hsv['v_low']}-{hsv['v_high']}]",
        f"FPS:{fps:5.1f}  {'[PAUSED]' if paused else '[TRACKING]' if tracking else '[SEARCHING]'}",
        "Q:quit  R:reset  S:save  L:load  SPACE:pause  P:pid",
    ]
    for i, line in enumerate(lines):
        _text(frame, line, 8, 18 + i * 20)

    if show_pid:
        pid_lines = [
            f"Pan  Kp={pid_cfg['pan_kp']:.4f} Ki={pid_cfg['pan_ki']:.5f} Kd={pid_cfg['pan_kd']:.4f}",
            f"Tilt Kp={pid_cfg['tilt_kp']:.4f} Ki={pid_cfg['tilt_ki']:.5f} Kd={pid_cfg['tilt_kd']:.4f}",
            f"Deadzone: {pid_cfg['deadzone']} px",
        ]
        for i, line in enumerate(pid_lines):
            _text(frame, line, 8, FRAME_H - 60 + i * 20, color=C_CYAN)

def draw_pid_bar(frame, label: str, value: float, max_val: float,
                 x: int, y: int, w: int = 120, h: int = 10):
    """Mini bar showing PID output magnitude."""
    ratio = min(abs(value) / max(max_val, 1e-6), 1.0)
    fill  = int(w * ratio)
    color = (0, 200, 80) if value >= 0 else (60, 80, 220)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    _text(frame, f"{label} {value:+.3f}", x + w + 6, y + h, scale=0.38)

# =============================================================
# Main
# =============================================================

def main():
    # --- Load config ---
    cfg = load_config()

    # --- Servo init ---
    print("[INFO] Initialising servos …")
    kit  = ServoKit(channels=16)
    pan  = float(PAN_START)
    tilt = float(TILT_START)
    kit.servo[PAN_CHANNEL].angle  = pan
    kit.servo[TILT_CHANNEL].angle = tilt
    time.sleep(0.5)

    # --- PID controllers ---
    #  Pan:  error = target_cx - CENTER_X
    #        positive error → target is right → need to decrease pan angle
    #        so PID output is SUBTRACTED from pan
    #  Tilt: error = target_cy - CENTER_Y
    #        positive error → target is below → need to increase tilt angle
    #        so PID output is ADDED to tilt

    pid_pan  = PID(cfg["pan_kp"],  cfg["pan_ki"],  cfg["pan_kd"],
                   out_min=-10, out_max=10)
    pid_tilt = PID(cfg["tilt_kp"], cfg["tilt_ki"], cfg["tilt_kd"],
                   out_min=-10, out_max=10)

    # --- Camera init ---
    print("[INFO] Starting camera …")
    picam2 = Picamera2()
    cam_cfg = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "BGR888"}
    )
    picam2.configure(cam_cfg)
    picam2.start()
    time.sleep(1.0)

    # --- Trackbars ---
    create_hsv_trackbars(cfg)
    create_pid_trackbars(cfg)

    paused   = False
    show_pid = True
    tracking = False
    prev_time = time.time()

    print("[INFO] Running — press Q to quit")

    try:
        while True:
            # FPS
            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Capture frame
            frame = picam2.capture_array()

            # Read trackbars
            hsv_cfg = read_hsv_trackbars()
            pid_cfg = read_pid_trackbars()
            cfg     = {**hsv_cfg, **pid_cfg}

            # Sync PID gains (live tuning)
            sync_pid_gains(pid_pan, pid_tilt, pid_cfg)
            deadzone = pid_cfg["deadzone"]

            # Build mask
            lower = np.array([hsv_cfg["h_low"],  hsv_cfg["s_low"],  hsv_cfg["v_low"]])
            upper = np.array([hsv_cfg["h_high"], hsv_cfg["s_high"], hsv_cfg["v_high"]])

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask      = cv2.inRange(hsv_frame, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            tracking    = False
            pan_output  = 0.0
            tilt_output = 0.0

            if not paused and contours:
                valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]

                if valid:
                    tracking = True
                    best     = max(valid, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(best)
                    cx = x + w // 2
                    cy = y + h // 2

                    draw_target(frame, x, y, w, h, cx, cy)

                    # Error in pixels
                    err_x = cx - CENTER_X   # + → right
                    err_y = cy - CENTER_Y   # + → below

                    # Apply deadzone
                    if abs(err_x) > deadzone:
                        pan_output = pid_pan.compute(err_x)
                        pan -= pan_output            # subtract: target right → pan left
                    else:
                        pid_pan.reset()

                    if abs(err_y) > deadzone:
                        tilt_output = pid_tilt.compute(err_y)
                        tilt += tilt_output          # add: target below → tilt down
                    else:
                        pid_tilt.reset()

                    pan  = max(PAN_MIN,  min(PAN_MAX,  pan))
                    tilt = max(TILT_MIN, min(TILT_MAX, tilt))

                    kit.servo[PAN_CHANNEL].angle  = pan
                    kit.servo[TILT_CHANNEL].angle = tilt

            else:
                # Not tracking — reset integrators so no wind-up
                if not tracking:
                    pid_pan.reset()
                    pid_tilt.reset()

            # Overlay
            draw_crosshair(frame, deadzone)
            draw_hud(frame, pan, tilt, hsv_cfg, pid_cfg, fps, paused, tracking, show_pid)

            # PID output bars (bottom-left)
            draw_pid_bar(frame, "Pan ",  pan_output,  10, 8, FRAME_H - 100)
            draw_pid_bar(frame, "Tilt",  tilt_output, 10, 8, FRAME_H - 85)

            # Mask thumbnail (top-right)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            thumb    = cv2.resize(mask_rgb, (160, 120))
            frame[0:120, FRAME_W - 160:FRAME_W] = thumb
            cv2.rectangle(frame, (FRAME_W - 160, 0), (FRAME_W, 120), (100, 100, 100), 1)

            cv2.imshow("Phase 2 — PID Tracking", frame)
            cv2.imshow(HSV_WIN, mask)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                pan, tilt = float(PAN_START), float(TILT_START)
                kit.servo[PAN_CHANNEL].angle  = pan
                kit.servo[TILT_CHANNEL].angle = tilt
                pid_pan.reset()
                pid_tilt.reset()
                print(f"[INFO] Reset → pan={PAN_START}° tilt={TILT_START}°")
            elif key == ord("s"):
                save_config(cfg)
            elif key == ord("l"):
                cfg = load_config()
                sync_pid_gains(pid_pan, pid_tilt, cfg)
                # Reload trackbars
                cv2.setTrackbarPos("H Low",  HSV_WIN, cfg["h_low"])
                cv2.setTrackbarPos("H High", HSV_WIN, cfg["h_high"])
                cv2.setTrackbarPos("S Low",  HSV_WIN, cfg["s_low"])
                cv2.setTrackbarPos("S High", HSV_WIN, cfg["s_high"])
                cv2.setTrackbarPos("V Low",  HSV_WIN, cfg["v_low"])
                cv2.setTrackbarPos("V High", HSV_WIN, cfg["v_high"])
                cv2.setTrackbarPos("Pan  Kp x1000",  PID_WIN, int(cfg["pan_kp"]  * 1000))
                cv2.setTrackbarPos("Pan  Kd x1000",  PID_WIN, int(cfg["pan_kd"]  * 1000))
                cv2.setTrackbarPos("Pan  Ki x10000", PID_WIN, int(cfg["pan_ki"]  * 10000))
                cv2.setTrackbarPos("Tilt Kp x1000",  PID_WIN, int(cfg["tilt_kp"] * 1000))
                cv2.setTrackbarPos("Tilt Kd x1000",  PID_WIN, int(cfg["tilt_kd"] * 1000))
                cv2.setTrackbarPos("Tilt Ki x10000", PID_WIN, int(cfg["tilt_ki"] * 10000))
                cv2.setTrackbarPos("Deadzone px",    PID_WIN, cfg["deadzone"])
            elif key == ord(" "):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord("p"):
                show_pid = not show_pid

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        print("[INFO] Cleaning up …")
        kit.servo[PAN_CHANNEL].angle  = PAN_START
        kit.servo[TILT_CHANNEL].angle = TILT_START
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Done")


if __name__ == "__main__":
    main()