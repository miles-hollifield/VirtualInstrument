# Virtual DJ Hand Controller

A computer vision-based application that allows you to control audio playback using hand gestures, turning your webcam into a virtual DJ controller.

## Features

- **Intuitive Gesture Controls**: Manipulate volume, speed, and pitch using natural hand movements
- **Real-time Audio Processing**: Instantly hear changes as you move your hands
- **Visual Feedback**: On-screen visualization of audio parameters and waveforms
- **Track Navigation**: Switch between tracks with hand twist gestures

![Virtual DJ Hand Controller Demo](docs/demo_screenshot.png)

## How It Works

The application uses your webcam to track hand movements through MediaPipe's hand landmark detection. Different gestures control various aspects of audio playback:

- **Volume**: Distance between left and right hands
- **Speed/Tempo**: Left hand pinch gesture (thumb to index finger)
- **Pitch**: Right hand pinch gesture (thumb to index finger)
- **Track Navigation**: Twist either hand to move forward/backward in the playlist

## Requirements

- Python 3.8+
- Webcam
- Audio files in WAV format

## Installation

1. Once you have the project installed on your machine, navigate to the parent directory of the project in your terminal.
2. Create and activate a virtual environment (recommended):
```
python -m venv venv
```
- On Windows
```
venv\Scripts\activate
```
- On macOS/Linux
```
source venv/bin/activate
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Create directories for your audio files:
```
mkdir -p songs sounds
```
5. Add your WAV audio files to the `songs` directory.

## Usage

1. Run the main application:
- python src/dj_pyo.py
2. Position yourself in front of the webcam, ensuring your hands are visible.

3. Control the audio with the following gestures:
- **Volume**: Move hands closer/further apart
- **Speed**: Pinch your left hand (thumb to index finger)
- **Pitch**: Pinch your right hand (thumb to index finger)
- **Previous Track**: Twist left hand
- **Next Track**: Twist right hand

4. Press 'Q' to quit, 'N' for next track, 'P' for previous track, and 'R' to reset parameters.

## Controls Overview

| Control | Gesture | Effect |
|---------|---------|--------|
| Volume | Distance between hands | 0.0 - 1.0 gain |
| Speed | Left hand pinch | 0.5x - 1.5x playback speed |
| Pitch | Right hand pinch | Adjusts pitch without affecting tempo |
| Next Track | Right hand twist | Moves to next track in playlist |
| Previous Track | Left hand twist | Returns to previous track in playlist |
| Reset | 'R' key | Returns all parameters to default values |

## Configuration

Audio files should be placed in the `songs` directory. The application supports WAV files by default.

## Troubleshooting

- **No camera detected**: Ensure your webcam is connected and not being used by another application
- **No hands detected**: Adjust lighting and position your hands clearly in the camera's view
- **Audio doesn't play**: Check that your audio files are in the correct format and location

## Development

The project is structured as follows:

- `src/dj_pyo.py`: Main application file
- `src/Trials/`: Alternative implementations and experimental features
- `songs/`: Directory for audio files
- `sounds/`: Directory for sound effects

## Acknowledgments

- MediaPipe for the hand tracking library
- Pyo for the audio processing engine
- OpenCV for camera capture and display