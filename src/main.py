import cv2
import mediapipe as mp
from gestures import detect_strum, detect_chord
from audio import play_chord

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

last_y = None
cooldown = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    chord_name = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            chord_name = detect_chord(hand_landmarks)

            if chord_name:
                cv2.putText(image, f"Chord: {chord_name}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                if detect_strum(last_y, wrist_y, cooldown):
                    print(f"Strum detected! Playing {chord_name}")
                    play_chord(chord_name)
                    cooldown = 20

            last_y = wrist_y

    if cooldown > 0:
        cooldown -= 1

    cv2.imshow("Gesture Guitar", image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()
