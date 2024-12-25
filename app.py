import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pyautogui  # For simulating mouse actions

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load the pre-trained model for gesture classification (gesture_model.h5)
model = tf.keras.models.load_model('gesture_model.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to extract hand landmarks from MediaPipe
def get_hand_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    landmarks = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the x, y, z coordinates for each of the 21 landmarks
            hand = []
            for landmark in hand_landmarks.landmark:
                hand.append([landmark.x, landmark.y, landmark.z])  # Store the 3D coordinates
            landmarks.append(hand)
    
    return landmarks

# Function to predict gesture using the model
def predict_gesture(landmarks):
    # Convert landmarks to a numpy array and flatten to match the model input shape
    landmarks = np.array(landmarks).flatten().reshape(1, -1)
    prediction = model.predict(landmarks)
    return np.argmax(prediction)  # Return the gesture class with the highest probability

# Function to perform left click (Pinch) and right click (Spread)
def perform_click(gesture_id):
    if gesture_id == 0:  # Pinch gesture - Left Click
        pyautogui.click()  # Perform Left Click
    elif gesture_id == 1:  # Spread gesture - Right Click
        pyautogui.rightClick()  # Perform Right Click
    elif gesture_id == 2:  # Swipe Up - Scroll Up
        pyautogui.scroll(10)  # Scroll up
    elif gesture_id == 3:  # Swipe Down - Scroll Down
        pyautogui.scroll(-10)  # Scroll down

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame to make it mirror-like

    # Get hand landmarks
    landmarks = get_hand_landmarks(frame)

    if landmarks:
        for hand in landmarks:
            # Make a gesture prediction using TensorFlow model
            gesture_id = predict_gesture(hand)
            
            # Perform left or right click based on gesture
            perform_click(gesture_id)
            
            # Display the predicted gesture
            if gesture_id == 0:
                gesture_text = "Gesture: Pinch (Left Click)"
            elif gesture_id == 1:
                gesture_text = "Gesture: Spread (Right Click)"
            elif gesture_id == 2:
                gesture_text = "Gesture: Swipe Up (Scroll Up)"
            elif gesture_id == 3:
                gesture_text = "Gesture: Swipe Down (Scroll Down)"
            else:
                gesture_text = "Gesture: Unknown"
            
            # Display the gesture label on the frame
            cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw the landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame with landmarks and gesture label
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()