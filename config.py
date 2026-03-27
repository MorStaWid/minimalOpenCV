"""
Shared configuration: constants, colours, thresholds, and MediaPipe aliases.
"""

import os

import mediapipe as mp

# ── MediaPipe Tasks API aliases ──────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ── MediaPipe drawing aliases ────────────────────────────────────────
draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks
DrawingSpec = mp.tasks.vision.drawing_utils.DrawingSpec
drawing_styles = mp.tasks.vision.drawing_styles
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections

# ── Colours (BGR) ────────────────────────────────────────────────────
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)

# ── Head-direction thresholds (pixels) ───────────────────────────────
YAW_THRESH = 15    # left / right
PITCH_THRESH = 10  # up / down

# ── Model paths (relative to this script's directory) ────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(SCRIPT_DIR, "models", "face_landmarker.task")
HAND_MODEL = os.path.join(SCRIPT_DIR, "models", "hand_landmarker.task")

# ── Quit-button layout ──────────────────────────────────────────────
QUIT_BTN_W = 90
QUIT_BTN_H = 36
QUIT_BTN_MARGIN = 16  # distance from top-right corner

RED_DARK = (0, 0, 160)
RED_BRIGHT = (60, 60, 230)
RED_HOVER = (80, 80, 255)

# ── Camera-switch button layout (to the left of QUIT) ───────────────
CAM_BTN_W = 100
CAM_BTN_H = 36
CAM_BTN_GAP = 10  # gap between CAM and QUIT buttons

BLUE_DARK = (160, 80, 0)
BLUE_BRIGHT = (210, 140, 40)
BLUE_HOVER = (240, 170, 60)

MAX_CAM_PROBE = 5  # how many camera indices to probe (0..4)
