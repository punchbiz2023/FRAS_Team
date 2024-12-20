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

# Function to save updated known faces dictionary to a file
def save_known_faces(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
    st.write(f"Updated known faces saved to {filename}")

# Preprocess image by converting it to RGB for face recognition
def preprocess_image(image_path):
    image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    return image_rgb

# Use MTCNN for more accurate face detection; returns face bounding boxes
def detect_faces_with_mtcnn(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    # Convert bounding boxes to (top, right, bottom, left) format
    locations = [(face['box'][1], face['box'][0] + face['box'][2],
                  face['box'][1] + face['box'][3], face['box'][0]) for face in faces]
    return locations

# Function to compute similarity between two face encodings using a combination of cosine similarity and Euclidean distance
def ensemble_similarity(face_encoding, known_encoding):
    # Cosine similarity calculation
    x = sum(a * b for a, b in zip(face_encoding, known_encoding))
    y = sqrt(sum(a ** 2 for a in face_encoding))
    z = sqrt(sum(b ** 2 for b in known_encoding))
    cosine = x / (y * z)

    # Euclidean distance calculation
    euclidean_distance = np.linalg.norm(np.array(face_encoding) - np.array(known_encoding))

    # Weighted average similarity score (adjust weights as needed)
    similarity_score = 0.6 * cosine + 0.4 * (1 / (1 + euclidean_distance))
    return similarity_score

# Function to mark attendance based on recognized names
def attendance(names, attendance_file):
    current_datetime = datetime.now().strftime("%d-%m-%Y %H:%M")

    # Read existing attendance records
    with open(attendance_file, 'r') as file:
        read = []
        reade = csv.reader(file)
        for i in reade:
            if i != []:
                read.append(i)

    # Display attendance summary
    total = len(read) - 1
    st.write("Total number of students : ", total)
    st.write("Number of present : ", len(names))
    st.write("Number of Absent : ", total - len(names))

    # Check if the current timestamp column exists, else add it
    if current_datetime not in read[0]:
        read[0].append(current_datetime)

    # Mark each student as present or absent
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

    # Write updated attendance back to the file
    with open(attendance_file, "w", newline="") as file:
        writ = csv.writer(file)
        writ.writerows(read)

# Function to recognize faces in the uploaded image, using known faces database
def recognize_faces(uploaded_file, known_faces):
    if uploaded_file is not None:
        # Load the uploaded image with PIL to retain quality
        image = Image.open(uploaded_file)

        # Convert image to OpenCV format for processing
        open_cv_image = np.array(image)
        image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Preprocess the image for face recognition
        unknown_image = preprocess_image(image)

        # Detect faces and get face encodings using MTCNN and face_recognition
        face_locations = detect_faces_with_mtcnn(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        # Scale factor for resizing the image display (optional)
        scale_percent = 50  # Resize percentage
        width = int(unknown_image.shape[1] * scale_percent / 100)
        height = int(unknown_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(unknown_image, dim, interpolation=cv2.INTER_AREA)

        # Adjust face locations for resized image
        adjusted_face_locations = [(int(top * scale_percent / 100),
                                    int(right * scale_percent / 100),
                                    int(bottom * scale_percent / 100),
                                    int(left * scale_percent / 100))
                                   for (top, right, bottom, left) in face_locations]

        # Iterate over each detected face and match with known faces
        for (top, right, bottom, left), face_encoding in zip(adjusted_face_locations, face_encodings):
            name = "UNKNOWN"
            best_similarity = -1

            # Compare face with each known face encoding
            for known_name, known_data in data.items():
                similarities = [ensemble_similarity(face_encoding, known_enc) for known_enc in
                                known_data['face_encodings']]
                max_similarity = max(similarities)

                # Select name with the highest similarity above the threshold
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    if max_similarity >= 0.81:
                        name = known_name

            # If name is not "UNKNOWN" and similarity is high, store new encoding
            if name != "UNKNOWN" and max_similarity >= 0.9:
                print(name)
                data[name]['face_encodings'].append(face_encoding)

            # Draw rectangle around detected face and label with name
            cv2.rectangle(resized_image, (left, top), (right, bottom), (0, 0, 255), 2)
            if "1" in name:
                name = name.replace("1", "").replace(" ", "")
            if name not in present_names:
                present_names.append(name)

            # Display name of the recognized person
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(resized_image, name, (left, bottom - 6), font, 0.6, (0, 0, 0), 1)

        save_known_faces(data, known_faces)  # Save any new encodings added

        return resized_image


# Function to process two uploaded images, combine results, and mark attendance
def divide_and_process(photo1, photo2, attendance_file, k_file):
    with st.status("Processing captured image...", expanded=True) as status:
        st.write("Recognizing Faces in image 1...")
        pro_image1 = recognize_faces(photo1, k_file)
        st.write("Recognizing Faces in image 2...")
        pro_image2 = recognize_faces(photo2, k_file)
        status.update(label="Processing Comleted!...", state="complete", expanded=False)
    if pro_image1 is not None and pro_image2 is not None:
        # Ensure both images are resized to the same height for stacking
        if pro_image1.shape[0] != pro_image2.shape[0]:
            pro_image2 = cv2.resize(pro_image2, (pro_image1.shape[1], pro_image1.shape[0]))

        # Combine processed images horizontally and display
        combined_image = np.hstack((pro_image1, pro_image2))
        st.image(combined_image, caption='Processed Image', use_column_width=True)
        st.write("Marking attendance")
        # Mark attendance and display records
        attendance(present_names, attendance_file)

        csv_path = attendance_file
        df = pd.read_csv(csv_path)
        st.write("### Attendance Records:")
        st.dataframe(df)

    else:
        if pro_image1 is not None:
            st.write("No faces detected in image 2")
            status.update(label="No faces detected in image 2", state="error", expanded=False)
            st.image(pro_image1, caption='Processed Image', use_column_width=True)
        elif pro_image2 is not None:
            st.write("No faces detected in image 1")
            status.update(label="No faces detected in image 1", state="error", expanded=False)
            st.image(pro_image2, caption='Processed Image', use_column_width=True)
        else:
            st.write("No faces detected or recognized")
            status.update(label="No faces detected or recognized", state="error", expanded=False)

# Streamlit page for image upload and attendance processing
def upload_image_page(known_faces_file, attendance_file):
    st.title("Upload Images for Attendance")
    global data
    global present_names
    present_names = []

    st.write(f"Using Known Faces File: {known_faces_file}")
    st.write(f"Using Attendance File: {attendance_file}")
    with open(known_faces_file, "rb") as f:
        data = pickle.load(f)

    # Image upload functionality for two images
    uploaded_file_1 = st.file_uploader("Choose Image 1...", type=["jpg", "jpeg", "png"])
    uploaded_file_2 = st.file_uploader("Choose Image 2...", type=["jpg", "jpeg", "png"])
    if st.button("Process!"):
        divide_and_process(uploaded_file_1, uploaded_file_2, attendance_file, known_faces_file)

    # Button to return to Class Selection Page
    def go_to_class_selection():
        st.session_state.page = "class_selection"

