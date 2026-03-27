# Head & Hand Movement Detector

Minimal OpenCV + MediaPipe project that detects **head direction** and **hand/finger movement** in real-time from your webcam.

## Features
- **Head pose detection** – shows whether you're looking Up / Down / Left / Right / Centre
- **Hand tracking** – tracks up to 2 hands with skeleton overlay
- **Finger counting** – displays how many fingers are raised
- **Movement arrows** – draws velocity arrows showing hand movement direction & speed
- **HUD overlay** – semi-transparent info box with all detection data

## Setup

```bash
# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
source venv/bin/activate
python head_hand_tracker.py          # default webcam
python head_hand_tracker.py --cam 1  # alternate camera
```

Press **q** to quit.

## How It Works

| Component | Library | Details |
|-----------|---------|---------|
| Head pose | MediaPipe Face Mesh | 468 face landmarks → nose-tip offset from face centre → direction label |
| Hand tracking | MediaPipe Hands | 21 landmarks per hand, skeleton drawing, wrist velocity arrows |
| Finger count | Custom heuristic | Compares finger-tip vs PIP joint y-coordinates |
| Display | OpenCV | Mirror view, landmark overlay, semi-transparent HUD |
