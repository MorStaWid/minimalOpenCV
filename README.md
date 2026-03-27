# Head & Hand Movement Detector

Minimal OpenCV + MediaPipe project that detects **head direction** and **hand/finger movement** in real-time from your webcam.

## Features
- **Head pose detection** – shows whether you're looking Up / Down / Left / Right / Centre
- **Hand tracking** – tracks up to 2 hands with skeleton overlay
- **Finger counting** – displays how many fingers are raised
- **Movement arrows** – draws velocity arrows showing hand movement direction & speed
- **HUD overlay** – semi-transparent info box with all detection data
- **Quit button** – clickable red QUIT button in the top-right corner

## Project Structure

```
minimalOpenCV/
├── head_hand_tracker.py   # Entry point – main loop & orchestration
├── config.py              # Constants, colours, thresholds, MediaPipe aliases
├── detectors.py           # Head direction & finger counting logic
├── drawing.py             # HUD, quit button, landmark rendering
├── requirements.txt       # Python dependencies
├── models/
│   ├── face_landmarker.task
│   └── hand_landmarker.task
└── venv/                  # Python virtual environment
```

## Setup

```bash
# Create & activate virtual environment (Python 3.12 recommended)
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

Press **q** or click the **QUIT** button to exit.

## How It Works

| Component | Module | Details |
|-----------|--------|---------|
| Head pose | `detectors.py` | 478 face landmarks → nose-tip offset from face centre → direction label |
| Hand tracking | `drawing.py` | 21 landmarks per hand, skeleton drawing, wrist velocity arrows |
| Finger count | `detectors.py` | Compares finger-tip vs PIP joint y-coordinates |
| Configuration | `config.py` | Colours, thresholds, model paths, MediaPipe API aliases |
| Display | `drawing.py` | Mirror view, landmark overlay, semi-transparent HUD, quit button |
