# VirtualInstrument

A computer vision-based project that allows you to play virtual instruments using hand gestures through your webcam.

## Instruments

### Virtual Guitar
Play guitar chords by forming hand shapes and strumming with your hand.

- Different finger positions represent different chords
- A downward motion of your wrist triggers a strum
- Visual feedback shows the detected chord

To run:
```
python src/main.py
```

### Virtual Trombone
Play a trombone by controlling the pitch with the distance between your hands.

- The distance between your hands controls the pitch (like a real trombone slide)
- Extend your index finger on either hand to play a note
- Visual feedback shows the current note and position

To run:
```
python src/trombone_main.py
```

## Requirements

Install the required packages:
```
pip install -r requirements.txt
```

## Controls

### Guitar
- Show one hand to the camera
- Form finger positions to select chords:
  - Index + Middle finger = C Major
  - Index + Middle + Ring finger = G Major
  - All four fingers extended = D Major
- Move your hand down quickly to strum
- ESC key to quit

### Trombone
- Show both hands to the camera
- Move hands closer or further apart to change pitch
- Extend the index finger on either hand to play the note
- ESC key to quit

## Directory Structure
```
VirtualInstrument/
├── src/
│   ├── main.py            # Guitar instrument main script
│   ├── trombone_main.py   # Trombone instrument main script
│   ├── gestures.py        # Hand gesture detection for guitar
│   ├── audio.py           # Audio playback for guitar
│   └── audio_trombone.py  # Audio synthesis for trombone
├── sounds/                # Sound files for guitar chords
├── requirements.txt       # Required packages
└── README.md              # This file
```