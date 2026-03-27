"""
Detection logic: head direction estimation and finger counting.
"""

from config import YAW_THRESH, PITCH_THRESH


def head_direction(landmarks, w, h):
    """Return (label, nose_point) based on nose-tip vs face centre.

    Args:
        landmarks: List of 478 face landmarks (normalised coords).
        w: Frame width in pixels.
        h: Frame height in pixels.

    Returns:
        (direction_label, nose_pixel_coords)  e.g. ("Up + Left", (320, 210))
    """
    nose = landmarks[1]       # nose tip
    forehead = landmarks[10]  # top-centre of face
    chin = landmarks[152]     # bottom-centre

    nose_pt = (int(nose.x * w), int(nose.y * h))
    mid_x = (landmarks[234].x + landmarks[454].x) / 2 * w  # left/right cheek
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


def fingers_up(landmarks):
    """Count how many fingers are extended (0-5, simple heuristic).

    Args:
        landmarks: List of 21 hand landmarks (normalised coords).

    Returns:
        Integer count of raised fingers.
    """
    tips = [8, 12, 16, 20]   # index, middle, ring, pinky tips
    pips = [6, 10, 14, 18]   # corresponding PIP joints
    count = 0

    for tip, pip_ in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip_].y:  # tip above PIP => finger up
            count += 1

    # Thumb: compare tip.x to IP.x (thumb extends sideways)
    if abs(landmarks[4].x - landmarks[3].x) > 0.04:
        count += 1

    return count
