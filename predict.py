

import cv2
import os
import numpy as np
from tensorflow import keras

# Load the trained sign language recognition model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model_1.h5'))
model = keras.models.load_model(model_path)

# Mapping of class labels to sign language gestures
class_labels = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',]  # Replace with your class labels

# Initialize sentence
sentence = ""

# Initialize flag to check if Enter key was pressed
enter_pressed = False

# Function to preprocess the input frame
def preprocess_frame(roi):
    # Preprocess the frame (resize, normalize, etc.)
    # Return the preprocessed frame
    resized_frame = cv2.resize(roi, (28, 28))
    grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    normalized_frame = grayscale_frame / 255.0
    return normalized_frame

# Function to perform sign language recognition
def recognize_sign_language(roi):
    preprocessed_frame = preprocess_frame(roi)
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    predicted_label = class_labels[np.argmax(prediction)]
    accuracy = np.max(prediction)
    return predicted_label, accuracy

# Function to update the sentence
def update_sentence(sign):
    global sentence
    sentence += sign 

# Function to process the video frames
def process_frames():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        

        # Define the region of interest (ROI) for sign language detection
        roi = frame[140:440, 140:440]

        # Perform sign language recognition
        predicted_sign, accuracy = recognize_sign_language(roi)

        # Display the predicted symbol and accuracy on the frame
        cv2.putText(frame, "Predicted: " + predicted_sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Accuracy: " + str(accuracy*100), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Sentence: " + sentence, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw a rectangle around the region of interest
        cv2.rectangle(frame, (140, 140), (340, 340), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)

        # Wait for key press
        key = cv2.waitKey(1)

        # Check if Enter key is pressed
        if key == 13:  # ASCII code for Enter key
            update_sentence(predicted_sign)

        # Exit the loop if '/' is pressed
        if key == ord('/'):
            break

    # Release the webcam and close the windows
    # Release the webcam and close the windows
    cap.release()
    cv2.destroyAllWindows()

# Start sign language recognition
process_frames() 
