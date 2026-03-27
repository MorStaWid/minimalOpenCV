"""
Head & Hand Movement Detector
==============================
Uses OpenCV + MediaPipe Tasks API to detect and visualise:
  - Head pose   (via FaceLandmarker, 478 landmarks)
  - Hand motion (via HandLandmarker, 21 landmarks per hand)

Run:
    python head_hand_tracker.py          # webcam 0
    python head_hand_tracker.py --cam 1  # webcam 1

Press 'q' or click the QUIT button to exit.
Click CAM to cycle through available cameras.
"""

import argparse
import os

import cv2
import mediapipe as mp
import numpy as np

from config import (
    FACE_MODEL,
    HAND_MODEL,
    MAGENTA,
    ORANGE,
    MAX_CAM_PROBE,
    BaseOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    VisionRunningMode,
)
from detectors import head_direction, fingers_up
from drawing import (
    draw_info_box,
    draw_quit_button,
    quit_button_rect,
    draw_cam_button,
    cam_button_rect,
    draw_hand_landmarks,
    draw_face_landmarks,
)


def _probe_cameras():
    """Find all available camera indices (0 .. MAX_CAM_PROBE-1).

    Returns a sorted list of indices that successfully open.
    """
    available = []
    for idx in range(MAX_CAM_PROBE):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available if available else [0]


def main():
    # ── Parse arguments ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Head & Hand Movement Detector")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    # ── Verify models exist ──────────────────────────────────────────
    for path, name in [(FACE_MODEL, "face_landmarker.task"),
                       (HAND_MODEL, "hand_landmarker.task")]:
        if not os.path.isfile(path):
            print(f"[ERROR] Model not found: {path}")
            print(f"  Download it with:")
            print(
                f"  curl -L -o models/{name} "
                f"https://storage.googleapis.com/mediapipe-models/"
                f"{'face_landmarker/face_landmarker' if 'face' in name else 'hand_landmarker/hand_landmarker'}"
                f"/float16/latest/{name}"
            )
            return

    # ── Probe available cameras ──────────────────────────────────────
    print("Probing cameras...", end=" ", flush=True)
    available_cams = _probe_cameras()
    print(f"found {len(available_cams)}: {available_cams}")

    # Use the requested cam if available, otherwise fall back to first found
    current_cam = args.cam if args.cam in available_cams else available_cams[0]

    # ── Open camera ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(current_cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {current_cam}")
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

    # ── Button click state ───────────────────────────────────────────
    quit_clicked = [False]
    cam_switch_requested = [False]
    mouse_pos = [0, 0]

    def on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            fw = param
            # Check QUIT button
            bx1, by1, bx2, by2 = quit_button_rect(fw)
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                quit_clicked[0] = True
            # Check CAM button
            cx1, cy1, cx2, cy2 = cam_button_rect(fw)
            if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                cam_switch_requested[0] = True

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

    prev_hand_positions = {}
    frame_ts = 0
    window_name = "Head & Hand Tracker"

    print(f"Head & Hand Tracker started (CAM {current_cam}) "
          f"– press 'q' or click QUIT to exit, click CAM to switch")

    mouse_cb_set = False

    # ── Main loop ────────────────────────────────────────────────────
    while cap.isOpened():
        # ── Handle camera switch ─────────────────────────────────────
        if cam_switch_requested[0]:
            cam_switch_requested[0] = False

            # Find the next camera in the available list
            idx = available_cams.index(current_cam) if current_cam in available_cams else -1
            next_idx = (idx + 1) % len(available_cams)
            next_cam = available_cams[next_idx]

            if next_cam != current_cam:
                # Release old camera and open new one
                cap.release()
                cap = cv2.VideoCapture(next_cam)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    current_cam = next_cam
                    # Clear stale detection results
                    latest_face_result[0] = None
                    latest_hand_result[0] = None
                    prev_hand_positions = {}
                    print(f"Switched to CAM {current_cam}")
                else:
                    # Failed to open, reopen the previous one
                    cap = cv2.VideoCapture(current_cam)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    print(f"[WARN] CAM {next_cam} unavailable, staying on CAM {current_cam}")

        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Register mouse callback once we know frame width
        if not mouse_cb_set:
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, on_mouse, w)
            mouse_cb_set = True

        # Convert to MediaPipe Image and send async
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        face_landmarker.detect_async(mp_image, frame_ts)
        hand_landmarker.detect_async(mp_image, frame_ts)

        # ── Process face results ─────────────────────────────────────
        head_info = []
        face_result = latest_face_result[0]
        if face_result and face_result.face_landmarks:
            for face_lms in face_result.face_landmarks:
                draw_face_landmarks(frame, face_lms)
                label, nose_pt = head_direction(face_lms, w, h)
                head_info.append(f"Head: {label}")
                cv2.circle(frame, nose_pt, 5, MAGENTA, -1)
        else:
            head_info.append("Head: not detected")

        # ── Process hand results ─────────────────────────────────────
        hand_info = []
        curr_hand_positions = {}
        hand_result = latest_hand_result[0]

        if hand_result and hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                draw_hand_landmarks(frame, hand_lms)

                # Determine handedness
                if hand_result.handedness and i < len(hand_result.handedness):
                    side = hand_result.handedness[i][0].category_name
                else:
                    side = "Unknown"

                fingers = fingers_up(hand_lms)
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
        all_lines = [f"=== Head & Hand Tracker (CAM {current_cam}) ==="] + head_info + hand_info
        draw_info_box(frame, all_lines, (10, 10))

        # ── CAM button (with hover effect) ───────────────────────────
        cx1, cy1, cx2, cy2 = cam_button_rect(w)
        cam_hovering = cx1 <= mouse_pos[0] <= cx2 and cy1 <= mouse_pos[1] <= cy2
        draw_cam_button(frame, current_cam, hover=cam_hovering)

        # ── QUIT button (with hover effect) ──────────────────────────
        bx1, by1, bx2, by2 = quit_button_rect(w)
        quit_hovering = bx1 <= mouse_pos[0] <= bx2 and by1 <= mouse_pos[1] <= by2
        draw_quit_button(frame, hover=quit_hovering)

        cv2.imshow(window_name, frame)

        # Exit on 'q' key OR quit-button click
        if cv2.waitKey(1) & 0xFF == ord("q") or quit_clicked[0]:
            break

    # ── Cleanup ──────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    hand_landmarker.close()


if __name__ == "__main__":
    main()
