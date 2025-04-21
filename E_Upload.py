import copy
import csv
import face_recognition
import numpy as np
import cv2
import pickle
import streamlit as st
import pandas as pd
from PIL import Image
from mtcnn import MTCNN
from math import sqrt
from datetime import datetime

# Function to save the dictionary of known faces into a file
def save_known_faces(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
    st.write(f"Updated known faces saved to {filename}")

# Preprocesses the image to convert it to RGB, necessary for face recognition compatibility
def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

# Uses MTCNN for face detection in the image, returning the face locations
def detect_faces_with_mtcnn(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    locations = [(face['box'][1], face['box'][0] + face['box'][2],
                  face['box'][1] + face['box'][3], face['box'][0]) for face in faces]
    return locations

# Calculates similarity score using a blend of cosine similarity and Euclidean distance
def ensemble_similarity(face_encoding, known_encoding):
    x = sum(a * b for a, b in zip(face_encoding, known_encoding))
    y = sqrt(sum(a ** 2 for a in face_encoding))
    z = sqrt(sum(b ** 2 for b in known_encoding))
    cosine = x / (y * z)
    euclidean_distance = np.linalg.norm(np.array(face_encoding) - np.array(known_encoding))
    similarity_score = 0.6 * cosine + 0.4 * (1 / (1 + euclidean_distance))
    return similarity_score

# Updates attendance based on recognized faces and logs them with timestamps
def attendance(names, attendance_file):
    current_datetime = datetime.now().strftime("%d-%m-%Y %H:%M")

    with open(attendance_file, 'r') as file:
        read = [row for row in csv.reader(file) if row]

    total = len(read) - 1
    st.write("Total number of students:", total)
    st.write("Number of present:", len(names))
    st.write("Number of absent:", total - len(names))

    if current_datetime not in read[0]:
        read[0].append(current_datetime)
    for i in range(1, len(read)):
        if read[i][0] in names and read[i][0] != "UNKNOWN":
            if len(read[i]) < len(read[0]):
                read[i].append("Present")
            else:
                read[i][-1] = "Present"
        else:
            if len(read[i]) < len(read[0]):
                read[i].append("Absent")
            else:
                read[i][-1] = "Absent"

    with open(attendance_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(read)

# Recognizes faces in an image, marks them, and updates known faces if similarity is high
def recognize_faces(image, known_faces_file):
    global data
    with open(known_faces_file, "rb") as f:
        data = pickle.load(f)

    unknown_image = preprocess_image(image)
    face_locations = detect_faces_with_mtcnn(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    scale_percent = 50
    width = int(unknown_image.shape[1] * scale_percent / 100)
    height = int(unknown_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(unknown_image, dim, interpolation=cv2.INTER_AREA)

    output_image = None
    adjusted_face_locations = [(int(top * scale_percent / 100),
                                int(right * scale_percent / 100),
                                int(bottom * scale_percent / 100),
                                int(left * scale_percent / 100))
                               for (top, right, bottom, left) in face_locations]

    for (top, right, bottom, left), face_encoding in zip(adjusted_face_locations, face_encodings):
        name = "UNKNOWN"
        best_similarity = -1
        for known_name, known_data in data.items():
            similarities = [ensemble_similarity(face_encoding, known_enc) for known_enc in known_data['face_encodings']]
            max_similarity = max(similarities)
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                if max_similarity >= 0.81:
                    name = known_name

        if name != "UNKNOWN" and max_similarity >= 0.9:
            data[name]['face_encodings'].append(face_encoding)

        cv2.rectangle(resized_image, (left, top), (right, bottom), (0, 0, 255), 2)
        if "1" in name:
            name = name.replace("1", "").replace(" ", "")
        if name not in present_names:
            present_names.append(name)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(resized_image, name, (left, bottom - 6), font, 0.6, (0, 0, 0), 1)

        output_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    save_known_faces(data, known_faces_file)
    return output_image

# Captures a photo from the specified IP camera URL
def capture_photo_from_url(ip_camera_url):
    camera = cv2.VideoCapture(ip_camera_url)
    st.write(f"Capturing image from {ip_camera_url}...")

    ret, image = camera.read()
    if ret:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        camera.release()
        return image_rgb
    else:
        st.write("Failed to capture image from:", ip_camera_url)
        camera.release()
        return None

# Main live capture page for marking attendance
def live_capture_page(known_faces_file, attendance_file):
    st.title("Live Attendance Capture - Dual Camera")
    st.write(f"Using Known Faces File: {known_faces_file}")
    st.write(f"Using Attendance File: {attendance_file}")

    global data
    global present_names
    present_names = []

    with open(known_faces_file, "rb") as f:
        data = pickle.load(f)

    ip_camera_url_1 = 'http://192.168.137.98:8080/video'
    ip_camera_url_2 = 'http://192.168.200.187:8080/video'

    if st.button('Capture from Both Cameras'):
        captured_image_1 = capture_photo_from_url(ip_camera_url_1)
        captured_image_2 = capture_photo_from_url(ip_camera_url_2)
        st.write("Image captured.")

        if captured_image_1 is not None and captured_image_2 is not None:
            st.image(captured_image_1, caption="Captured Image from Camera 1", use_column_width=True)
            st.image(captured_image_2, caption="Captured Image from Camera 2", use_column_width=True)
            output_image_1 = recognize_faces(captured_image_1, known_faces_file)
            output_image_2 = recognize_faces(captured_image_2, known_faces_file)
            if output_image_1 is not None and output_image_2 is not None:
                if output_image_1.shape[0] != output_image_2.shape[0]:
                    pro_image2 = cv2.resize(output_image_2, (output_image_1.shape[1], output_image_1.shape[0]))

                # Combine processed images horizontally and display
                combined_image = np.hstack((output_image_1, output_image_2))
                st.image(combined_image, caption='Processed Image', use_column_width=True)
            else:
                if output_image_1 is not None:
                    st.image(output_image_1, caption='Processed Image', use_column_width=True)
                    st.write("No faces detected in Camera 2 image.")
                elif output_image_2 is not None:
                    st.image(output_image_2, caption='Processed Image', use_column_width=True)
                    st.write("No faces detected in Camera 1 image.")
                else:
                    st.write("No faces detected")


            attendance(present_names, attendance_file)
            df = pd.read_csv(attendance_file)
            st.write("### Attendance Records:")
            st.dataframe(df)

