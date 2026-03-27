"""
Head & Hand Movement Detector
==============================
Uses OpenCV + MediaPipe Tasks API to detect and visualise:
  • Head pose   – via FaceLandmarker (478 landmarks)
  • Hand motion – via HandLandmarker (21 landmarks per hand)

Run:
    python head_hand_tracker.py          # webcam 0
    python head_hand_tracker.py --cam 1  # webcam 1

Press 'q' to quit.
"""

import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe Tasks API aliases ──────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Drawing helpers
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

# ── Head‑direction thresholds (pixels) ───────────────────────────────
YAW_THRESH = 15   # left / right
PITCH_THRESH = 10  # up / down

# ── Model paths (relative to this script) ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(SCRIPT_DIR, "models", "face_landmarker.task")
HAND_MODEL = os.path.join(SCRIPT_DIR, "models", "hand_landmarker.task")


def _head_direction(landmarks, w, h):
    """Return (label, nose_point) based on nose‑tip vs face centre."""
    nose = landmarks[1]          # nose tip
    forehead = landmarks[10]     # top‑centre of face
    chin = landmarks[152]        # bottom‑centre

    nose_pt = (int(nose.x * w), int(nose.y * h))
    mid_x = (landmarks[234].x + landmarks[454].x) / 2 * w   # left/right cheek
    mid_y = (forehead.y + chin.y) / 2 * h

    dx = nose_pt[0] - mid_x
    dy = nose_pt[1] - mid_y

    parts = []
    if dy < -PITCH_THRESH:
        parts.append("Up")
    elif dy > PITCH_THRESH:
        parts.append("Down")
    if dx < -YAW_THRESH:
        parts.append("Left")
    elif dx > YAW_THRESH:
        parts.append("Right")

    label = " + ".join(parts) if parts else "Centre"
    return label, nose_pt


def _fingers_up(landmarks):
    """Count how many fingers are extended (simple heuristic)."""
    tips = [8, 12, 16, 20]        # index, middle, ring, pinky tips
    pips = [6, 10, 14, 18]        # corresponding PIP joints
    count = 0
    for tip, pip_ in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip_].y:   # tip above PIP => finger up
            count += 1
    # Thumb: compare tip.x to IP.x
    if abs(landmarks[4].x - landmarks[3].x) > 0.04:
        count += 1
    return count


def _draw_info_box(frame, lines, origin, bg=(40, 40, 40), alpha=0.7):
    """Draw a semi‑transparent box with text lines."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    padding = 8
    line_h = 22

    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_w = max(max_w, tw)

    x, y = origin
    box_w = max_w + padding * 2
    box_h = len(lines) * line_h + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), bg, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, line in enumerate(lines):
        ty = y + padding + (i + 1) * line_h
        cv2.putText(frame, line, (x + padding, ty), font, scale, WHITE, thickness, cv2.LINE_AA)


# ── Quit‑button constants ────────────────────────────────────────────
QUIT_BTN_W = 90
QUIT_BTN_H = 36
QUIT_BTN_MARGIN = 16          # distance from top‑right corner

RED_DARK = (0, 0, 160)
RED_BRIGHT = (60, 60, 230)
RED_HOVER = (80, 80, 255)


def _quit_button_rect(frame_w):
    """Return (x1, y1, x2, y2) for the quit button in the top‑right."""
    x1 = frame_w - QUIT_BTN_W - QUIT_BTN_MARGIN
    y1 = QUIT_BTN_MARGIN
    x2 = x1 + QUIT_BTN_W
    y2 = y1 + QUIT_BTN_H
    return x1, y1, x2, y2


def _draw_quit_button(frame, hover=False):
    """Draw a styled QUIT button in the top‑right corner."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _quit_button_rect(w)

    # Button background with slight transparency
    overlay = frame.copy()
    bg_colour = RED_HOVER if hover else RED_BRIGHT
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_colour, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), RED_DARK, 2)

    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = "QUIT"
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx = x1 + (QUIT_BTN_W - tw) // 2
    ty = y1 + (QUIT_BTN_H + th) // 2
    cv2.putText(frame, label, (tx, ty), font, scale, WHITE, thickness, cv2.LINE_AA)


def _draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand skeleton on the frame using the new Tasks API."""
    draw_landmarks(
        frame,
        hand_landmarks,
        HandLandmarksConnections.HAND_CONNECTIONS,
        drawing_styles.get_default_hand_landmarks_style(),
        drawing_styles.get_default_hand_connections_style(),
    )


def _draw_face_landmarks(frame, face_landmarks):
    """Draw face mesh tesselation on the frame using the new Tasks API."""
    draw_landmarks(
        frame,
        face_landmarks,
        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        drawing_styles.get_default_face_mesh_tesselation_style(),
    )


def main():
    parser = argparse.ArgumentParser(description="Head & Hand Movement Detector")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    # Verify models exist
    for path, name in [(FACE_MODEL, "face_landmarker.task"), (HAND_MODEL, "hand_landmarker.task")]:
        if not os.path.isfile(path):
            print(f"[ERROR] Model not found: {path}")
            print(f"  Download it with:")
            print(f"  curl -L -o models/{name} https://storage.googleapis.com/mediapipe-models/{'face_landmarker/face_landmarker' if 'face' in name else 'hand_landmarker/hand_landmarker'}/float16/latest/{name}")
            return

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Shared state for async callbacks ─────────────────────────────
    latest_face_result = [None]
    latest_hand_result = [None]

    def on_face_result(result, output_image, timestamp_ms):
        latest_face_result[0] = result

    def on_hand_result(result, output_image, timestamp_ms):
        latest_hand_result[0] = result

    # ── Quit‑button click state ──────────────────────────────────────
    quit_clicked = [False]
    mouse_pos = [0, 0]

    def on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            fw = param  # frame width passed via setMouseCallback
            bx1, by1, bx2, by2 = _quit_button_rect(fw)
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                quit_clicked[0] = True

    # ── Create landmarkers (LIVE_STREAM mode) ────────────────────────
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_face_result,
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_hand_result,
    )

    face_landmarker = FaceLandmarker.create_from_options(face_options)
    hand_landmarker = HandLandmarker.create_from_options(hand_options)

    prev_hand_positions = {}  # track wrist positions for velocity arrows
    frame_ts = 0
    window_name = "Head & Hand Tracker"

    print("Head & Hand Tracker started – press 'q' or click QUIT to exit")

    # We'll set the mouse callback after the first frame so we know the width
    mouse_cb_set = False

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Register mouse callback once we know frame width
        if not mouse_cb_set:
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, on_mouse, w)
            mouse_cb_set = True

        # Convert to MediaPipe Image and send async
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33  # ~30 fps timestamps (must be monotonically increasing)

        face_landmarker.detect_async(mp_image, frame_ts)
        hand_landmarker.detect_async(mp_image, frame_ts)

        # ── Face results ─────────────────────────────────────────────
        head_info = []
        face_result = latest_face_result[0]
        if face_result and face_result.face_landmarks:
            for face_lms in face_result.face_landmarks:
                _draw_face_landmarks(frame, face_lms)
                label, nose_pt = _head_direction(face_lms, w, h)
                head_info.append(f"Head: {label}")
                cv2.circle(frame, nose_pt, 5, MAGENTA, -1)
        else:
            head_info.append("Head: not detected")

        # ── Hand results ─────────────────────────────────────────────
        hand_info = []
        curr_hand_positions = {}
        hand_result = latest_hand_result[0]

        if hand_result and hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                _draw_hand_landmarks(frame, hand_lms)

                # Determine handedness
                if hand_result.handedness and i < len(hand_result.handedness):
                    side = hand_result.handedness[i][0].category_name
                else:
                    side = "Unknown"

                fingers = _fingers_up(hand_lms)
                wrist = hand_lms[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                curr_hand_positions[side] = (wx, wy)

                hand_info.append(f"{side} hand: {fingers} finger(s) up")

                # Movement arrow (velocity)
                if side in prev_hand_positions:
                    px, py = prev_hand_positions[side]
                    dx, dy = wx - px, wy - py
                    speed = np.hypot(dx, dy)
                    if speed > 5:
                        end_pt = (wx + dx * 3, wy + dy * 3)
                        cv2.arrowedLine(frame, (wx, wy), end_pt, ORANGE, 2, tipLength=0.3)
                        hand_info[-1] += f"  | moving ({int(speed)} px/f)"
        else:
            hand_info.append("Hands: not detected")

        prev_hand_positions = curr_hand_positions

        # ── HUD ──────────────────────────────────────────────────────
        all_lines = ["=== Head & Hand Tracker ==="] + head_info + hand_info
        _draw_info_box(frame, all_lines, (10, 10))

        # ── Quit button (with hover effect) ──────────────────────────
        bx1, by1, bx2, by2 = _quit_button_rect(w)
        hovering = bx1 <= mouse_pos[0] <= bx2 and by1 <= mouse_pos[1] <= by2
        _draw_quit_button(frame, hover=hovering)

        cv2.imshow(window_name, frame)

        # Exit on 'q' key OR quit‑button click
        if cv2.waitKey(1) & 0xFF == ord("q") or quit_clicked[0]:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    hand_landmarker.close()


if __name__ == "__main__":
    main()
