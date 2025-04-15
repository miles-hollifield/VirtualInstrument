import os
import cv2
import numpy as np
import mediapipe as mp
import math
from pyo import *

player = None
hands_detected = False
pitch_shifter = None  # Added for pitch control

# === Setup Pyo Server ===
s = Server().boot()
s.start()

# === Playlist ===
song_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../songs'))
song_files = [os.path.join(song_dir, f) for f in os.listdir(song_dir) if f.endswith('.wav')]
current_track_index = 0

# === Load Track Function (doesn't play immediately) ===
def load_track(index):
    global player, pitch_shifter
    if player is not None:
        player.stop()
    
    # Create player with initial settings
    player = SfPlayer(song_files[index], speed=1, loop=True, mul=0.0)
    
    # Create pitch shifter using a better approach for pitch-only changes
    # Use the Harmonizer with appropriate settings to modify pitch without affecting speed
    pitch_shifter = Harmonizer(player, transpo=0, feedback=0, winsize=0.1, mul=1).out()
    
    # Don't output player directly, only through the pitch shifter
    # This prevents the original signal from being heard
    # player.out()

load_track(current_track_index)

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.6,
                       max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# === Control Parameters ===
volume = 0.5
speed = 1.0
pitch_transpo = 0  # Transposition value in semitones

# === Utility Functions ===
def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_pinch_distance(hand):
    thumb = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return distance(thumb, index)

# Function to convert Hz to semitones (relative to A4 = 440Hz)
def hz_to_transpo(hz):
    # Convert Hz to semitones relative to A4 (440Hz)
    return 12 * math.log2(hz / 440.0)

# === Webcam ===
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Hand DJ (Pyo) Started - [Q]uit, [N]ext, [P]revious, [R]eset")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    left_hand = None
    right_hand = None
    if results.multi_hand_landmarks and results.multi_handedness:
        hands_detected = True
        for info, landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            label = info.classification[0].label
            if label == "Left":
                left_hand = landmarks
            elif label == "Right":
                right_hand = landmarks

    # Play when hands are first detected
    if hands_detected and player.mul == 0.0:
        player.setMul(float(volume))

    # Volume - distance between pinch centers (scaled to 0-10)
    if left_hand and right_hand:
        # Get pinch points
        left_thumb = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        left_index = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        right_thumb = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        right_index = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate center of each pinch
        left_pinch_center = np.array([(left_thumb.x + left_index.x) / 2, (left_thumb.y + left_index.y) / 2])
        right_pinch_center = np.array([(right_thumb.x + right_index.x) / 2, (right_thumb.y + right_index.y) / 2])
        
        # Calculate distance between pinch centers
        pinch_dist = np.linalg.norm(right_pinch_center - left_pinch_center) * frame_width
        
        # More sensitive volume mapping - reaches max volume with less distance
        # Adjust these values to fine-tune sensitivity
        min_dist = 50  # Distance (in pixels) for minimum volume
        max_dist = 300  # Distance (in pixels) for maximum volume
        
        # Linear mapping from distance to volume range
        volume_display = float(min(10.0, max(0.0, 
            10.0 * (pinch_dist - min_dist) / (max_dist - min_dist)
        )))
        
        volume = volume_display / 10.0  # Convert to 0-1 range for player
        
        if pitch_shifter:
            pitch_shifter.setMul(volume)  # Set volume on the pitch shifter

    # Speed - left pinch
    if left_hand:
        pinch = get_pinch_distance(left_hand)
        speed = float(max(0.1, min(2.0, 2.0 - pinch * 4)))
        player.setSpeed(speed)

    # Pitch - right pinch (implemented using Harmonizer)
    if right_hand and pitch_shifter:
        pinch = get_pinch_distance(right_hand)
        
        # Calculate pitch in Hz (20Hz to 600Hz)
        pitch_hz = max(20, min(600, 600 - pinch * 1200))
        
        # Convert Hz to semitones for transposition
        # For Harmonizer, we need semitones relative to original pitch
        pitch_transpo = hz_to_transpo(pitch_hz) - hz_to_transpo(440)
        
        # Apply transposition to the Harmonizer without affecting speed
        pitch_shifter.setTranspo(pitch_transpo)

    # Don't draw full hand landmarks
    # for hand in results.multi_hand_landmarks or []:
    #     mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    
    # Draw visual indicators for pinch and pinch center distance
    if left_hand and right_hand:
        # Get pinch points
        left_thumb = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        left_index = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        right_thumb = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        right_index = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Convert normalized coordinates to pixel values
        left_thumb_px = (int(left_thumb.x * frame_width), int(left_thumb.y * frame_height))
        left_index_px = (int(left_index.x * frame_width), int(left_index.y * frame_height))
        right_thumb_px = (int(right_thumb.x * frame_width), int(right_thumb.y * frame_height))
        right_index_px = (int(right_index.x * frame_width), int(right_index.y * frame_height))
        
        # Calculate center of pinch lines
        left_pinch_center = (
            (left_thumb_px[0] + left_index_px[0]) // 2,
            (left_thumb_px[1] + left_index_px[1]) // 2
        )
        right_pinch_center = (
            (right_thumb_px[0] + right_index_px[0]) // 2,
            (right_thumb_px[1] + right_index_px[1]) // 2
        )
        
        # Draw lines between pinch points
        cv2.line(frame, left_thumb_px, left_index_px, (0, 255, 0), 2)  # Green line for left hand pinch
        cv2.line(frame, right_thumb_px, right_index_px, (0, 0, 255), 2)  # Red line for right hand pinch
        
        # Draw dots at center of pinch lines
        cv2.circle(frame, left_pinch_center, 8, (255, 255, 0), -1)  # Yellow dot for left pinch center
        cv2.circle(frame, right_pinch_center, 8, (255, 255, 0), -1)  # Yellow dot for right pinch center
        
        # Draw line between pinch centers
        cv2.line(frame, left_pinch_center, right_pinch_center, (255, 255, 255), 2)  # White line for volume control
        
        # Calculate distance between pinch centers (for volume control)
        pinch_center_distance = np.sqrt(
            (right_pinch_center[0] - left_pinch_center[0])**2 + 
            (right_pinch_center[1] - left_pinch_center[1])**2
        )
        
        # Display the distance between pinch centers and volume level
        midpoint = (
            (left_pinch_center[0] + right_pinch_center[0]) // 2,
            (left_pinch_center[1] + right_pinch_center[1]) // 2
        )
        
        cv2.putText(frame, f"Dist: {pinch_dist:.1f}px", 
                   (midpoint[0] - 45, midpoint[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Vol: {volume_display:.1f}/10", 
                   (midpoint[0] - 45, midpoint[1] + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display values with proper scaling
    volume_display = volume * 10.0  # Convert back to 0-10 scale for display
    
    cv2.putText(frame, f"Volume: {volume_display:.2f}/10", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Speed: {speed:.2f}x", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Pitch: {pitch_hz if 'pitch_hz' in locals() else 220:.0f}Hz", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Track: {os.path.basename(song_files[current_track_index])}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
    cv2.putText(frame, "Controls:", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1)
    cv2.putText(frame, "- Left pinch: Speed | Right pinch: Pitch | Hand distance: Volume", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
    cv2.putText(frame, "- Q: Quit | R: Reset | N: Next | P: Previous", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

    cv2.imshow("DJ Controller (Pyo)", frame)
    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        current_track_index = (current_track_index + 1) % len(song_files)
        load_track(current_track_index)
        hands_detected = False
    elif key == ord('p'):
        current_track_index = (current_track_index - 1) % len(song_files)
        load_track(current_track_index)
        hands_detected = False
    elif key == ord('r'):
        volume = 0.5
        volume_display = 5.0
        speed = 1.0
        pitch_hz = 440
        pitch_transpo = 0
        player.setMul(float(volume))
        player.setSpeed(float(speed))
        if pitch_shifter:
            pitch_shifter.setTranspo(pitch_transpo)

cap.release()
cv2.destroyAllWindows()
s.stop()
s.shutdown()
print("Exited cleanly.")