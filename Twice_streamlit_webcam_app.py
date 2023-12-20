import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Constants
IMAGE_SIZE = (160, 160)

# Predefined names of TWICE members corresponding to the label indices
members = ["Chaeyoung", "Dahyun", "Jeongyeon", "Jihyo", "Mina", "Momo", "Nayeon", "Sana", "Tzuyu"]

# Load the pre-trained models
face_recognition_model = keras.models.load_model('twice_face_recognition_model_new.h5')
face_detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def process_frame(frame):
    # Create a dictionary to store random colors for recognized names
    name_to_color = {}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detection_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces and recognize members
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, IMAGE_SIZE)
        face_img = face_img / 255.0  # Normalize pixel values

        # Make predictions using your recognition model
        predictions = face_recognition_model.predict(np.array([face_img]))
        predicted_label = np.argmax(predictions[0])

        # Get the TWICE member name based on the predicted label
        predicted_member_name = members[predicted_label]

        # Get the confidence score (classification score) and convert to percentage
        confidence_score = predictions[0][predicted_label] * 100

        # Generate a random color for the name or reuse the same color if already assigned
        color = name_to_color.get(predicted_member_name)
        if color is None:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            name_to_color[predicted_member_name] = color

        # Annotate the frame with a colored rectangle, member's name, and confidence score
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_position = (x, y - 10)  # Display the name above the face
        text = f"{predicted_member_name} ({confidence_score:.2f}%)"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, text, text_position, font, 0.5, color, thickness, cv2.LINE_AA)

    return frame

st.title('Twice Camera Facial Recognition App')

# Custom photo header
header_image_url = 'https://i.pinimg.com/736x/b9/31/5e/b9315eab7b11ec550295dda5cc72a2be.jpg'  # Replace with the URL of your image or local path
st.image(header_image_url, use_column_width=True)

run = st.checkbox('Click for Run Webcam')
frame_window = st.empty()

# Webcam loop
cap = cv2.VideoCapture(0)
while run:
    ret, frame = cap.read()
    if not ret:
        break

    # Process each frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    frame_window.image(processed_frame, channels="BGR", use_column_width=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not run:
    st.write("The Cam is Stopped")

# Copyright notice
st.markdown('---')  
st.markdown('* Made by Bondan Vitto © 2023') 

cap.release()
