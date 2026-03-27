"""
Drawing utilities: HUD overlay, quit button, and MediaPipe landmark rendering.
"""

import cv2

from config import (
    WHITE,
    QUIT_BTN_W,
    QUIT_BTN_H,
    QUIT_BTN_MARGIN,
    RED_DARK,
    RED_BRIGHT,
    RED_HOVER,
    CAM_BTN_W,
    CAM_BTN_H,
    CAM_BTN_GAP,
    BLUE_DARK,
    BLUE_BRIGHT,
    BLUE_HOVER,
    draw_landmarks,
    drawing_styles,
    FaceLandmarksConnections,
    HandLandmarksConnections,
)


# ── HUD info box ────────────────────────────────────────────────────

def draw_info_box(frame, lines, origin, bg=(40, 40, 40), alpha=0.7):
    """Draw a semi-transparent box with text lines.

    Args:
        frame:  The video frame to draw on (modified in place).
        lines:  List of strings to display.
        origin: (x, y) top-left corner of the box.
        bg:     Background colour (BGR).
        alpha:  Opacity of the background (0.0–1.0).
    """
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
        cv2.putText(
            frame, line, (x + padding, ty),
            font, scale, WHITE, thickness, cv2.LINE_AA,
        )


# ── Quit button ─────────────────────────────────────────────────────

def quit_button_rect(frame_w):
    """Return (x1, y1, x2, y2) for the quit button in the top-right."""
    x1 = frame_w - QUIT_BTN_W - QUIT_BTN_MARGIN
    y1 = QUIT_BTN_MARGIN
    x2 = x1 + QUIT_BTN_W
    y2 = y1 + QUIT_BTN_H
    return x1, y1, x2, y2


def draw_quit_button(frame, hover=False):
    """Draw a styled QUIT button in the top-right corner.

    Args:
        frame: The video frame to draw on (modified in place).
        hover: If True, use the brighter hover colour.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = quit_button_rect(w)

    # Semi-transparent background
    overlay = frame.copy()
    bg_colour = RED_HOVER if hover else RED_BRIGHT
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_colour, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), RED_DARK, 2)

    # Centred label
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = "QUIT"
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx = x1 + (QUIT_BTN_W - tw) // 2
    ty = y1 + (QUIT_BTN_H + th) // 2
    cv2.putText(frame, label, (tx, ty), font, scale, WHITE, thickness, cv2.LINE_AA)


# ── Camera-switch button ────────────────────────────────────────────

def cam_button_rect(frame_w):
    """Return (x1, y1, x2, y2) for the CAM button, left of the QUIT button."""
    quit_x1 = frame_w - QUIT_BTN_W - QUIT_BTN_MARGIN
    x2 = quit_x1 - CAM_BTN_GAP
    x1 = x2 - CAM_BTN_W
    y1 = QUIT_BTN_MARGIN
    y2 = y1 + CAM_BTN_H
    return x1, y1, x2, y2


def draw_cam_button(frame, cam_index, hover=False):
    """Draw a styled camera-switch button showing the current camera index.

    Args:
        frame:     The video frame to draw on (modified in place).
        cam_index: Current camera index to display.
        hover:     If True, use the brighter hover colour.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = cam_button_rect(w)

    # Semi-transparent background
    overlay = frame.copy()
    bg_colour = BLUE_HOVER if hover else BLUE_BRIGHT
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_colour, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE_DARK, 2)

    # Centred label
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"CAM {cam_index}"
    scale, thickness = 0.55, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx = x1 + (CAM_BTN_W - tw) // 2
    ty = y1 + (CAM_BTN_H + th) // 2
    cv2.putText(frame, label, (tx, ty), font, scale, WHITE, thickness, cv2.LINE_AA)


# ── MediaPipe landmark rendering ────────────────────────────────────

def draw_hand_landmarks(frame, hand_landmarks):
    """Draw the 21-point hand skeleton on the frame."""
    draw_landmarks(
        frame,
        hand_landmarks,
        HandLandmarksConnections.HAND_CONNECTIONS,
        drawing_styles.get_default_hand_landmarks_style(),
        drawing_styles.get_default_hand_connections_style(),
    )


def draw_face_landmarks(frame, face_landmarks):
    """Draw face mesh tessellation on the frame."""
    draw_landmarks(
        frame,
        face_landmarks,
        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        drawing_styles.get_default_face_mesh_tesselation_style(),
    )
