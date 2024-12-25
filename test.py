import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Set paths for your dataset (adjust paths as needed)
train_data_path = 'train.csv'  # Replace with your train data path
test_data_path = 'test.csv'    # Replace with your test data path

# Load the dataset
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Extract the hand landmarks using MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to extract landmarks from an 
def extract_landmarks(image):
    if image is None:
        return np.zeros(63)  # Return a vector of zeros if image is not loaded correctly 
    
    # Ensure image is in uint8 format before passing it to cv2.cvtColor
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark  # Assuming the first hand
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()
    
    return np.zeros(63)  # Return a vector of zeros if no landmarks are detected
 # Return a vector of zeros if no landmarks are detected

# Function to preprocess image (resize and normalize)
def preprocess_image(image):
    if image is None:
        return None  # Return None if image is not loaded correctly
    image_resized = cv2.resize(image, (471, 665))  # Resize to (471, 665)
    image_normalized = image_resized / 255.0  # Normalize pixel values to [0, 1]
    return image_normalized

# Prepare training and test data (hand landmarks)
X_train = []
y_train = []
X_test = []
y_test = []

# Extract landmarks from training data images
for i in range(len(train_data)):
    image = train_data.drop('label', axis=1).iloc[i].values.reshape(28, 28)  # Reshape to 28x28
    image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel (RGB)
    landmarks = extract_landmarks(image)
    X_train.append(landmarks)
    y_train.append(train_data['label'].iloc[i])

# Extract landmarks from test data images
for i in range(len(test_data)):
    image = test_data.drop('label', axis=1).iloc[i].values.reshape(28, 28)  # Reshape to 28x28
    image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel (RGB)
    landmarks = extract_landmarks(image)
    X_test.append(landmarks)
    y_test.append(test_data['label'].iloc[i])

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalize the data (optional, since it's already normalized)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, 26)  # 26 classes (A-Z)
y_test = to_categorical(y_test, 26)

# Define the model
model = models.Sequential([
    layers.InputLayer(input_shape=(63,)),  # 63 input features (landmarks)
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(26, activation='softmax')  # 26 output neurons for A-Z
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Save the trained model for later use
model.save('gesture_model.h5')