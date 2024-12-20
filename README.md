
# FRASbiz: Face Recognition Attendance System

**FRASbiz** is a face recognition-based attendance management system. Using **MTCNN** for face detection and the **face_recognition** library for face matching, this system automatically identifies individuals and logs attendance for each session. This project can be used for live capturing or from an uploaded photo, simplifying attendance tracking in classrooms, offices, or other group environments.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Files](#configuration-files)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Face Detection and Recognition**: Detects and identifies faces using a combination of MTCNN and cosine similarity-based matching.
- **Ensemble Similarity Measure**: Uses a hybrid approach combining cosine similarity and Euclidean distance for accurate face matching.
- **Attendance Logging**: Records attendance and appends data to a CSV file, marking individuals as present or absent.
- **Streamlit Interface**: Interactive, web-based GUI for ease of use.
- **Real-Time and Batch Modes**: Supports both real-time capture from a webcam and batch processing of uploaded images.

## Tech Stack
- **Python**: Core programming language for development.
- **Streamlit**: For the web-based user interface.
- **OpenCV**: Image processing and real-time video capture.
- **MTCNN**: For high-accuracy face detection.
- **face_recognition**: For face encoding and matching.
- **Pandas**: For handling attendance records.
- **Pickle**: Serialization of known faces data.

## Installation

### Prerequisites
1. **Python 3.7+**: Ensure Python is installed on your machine.
2. **Virtual Environment (Recommended)**: Use a virtual environment to manage dependencies.

### Steps
```bash
# Clone the repository:
git clone https://github.com/Punchbiz-interns/FRASbiz.git
cd FRASbiz

# Install dependencies:
pip install -r requirements.txt
```

Ensure you have access to a webcam or a camera for live capture mode.

## Usage
Start the application by running:
```bash
streamlit run FRASbiz.py
```

### Live Capture Mode
1. Navigate to the **Live Attendance Capture** section.
2. Click **Capture** to take a photo from the webcam.
3. The system will automatically detect and identify faces, display bounding boxes, and log attendance.
4. Attendance data is saved to the specified CSV file.

### Upload Photo Mode
1. Upload a group photo for attendance logging.
2. The system will process the image, identify known faces, and log attendance accordingly.
3. Attendance data is updated in real-time and displayed in the interface.

## Configuration Files
- **known_faces.pkl**: Stores face encodings of known individuals.
- **attendance.csv**: Logs attendance records with timestamp, names, and status (Present/Absent).

## File Structure
```plaintext
FRASbiz/
├── FRASbiz.py                     # Main Streamlit application
├── models/
│   ├── known_faces.pkl        # Known faces file
├── utils/
│   ├── face_detection.py      # Helper functions for face detection
│   ├── attendance.py          # Attendance tracking logic
├── data/
│   ├── attendance.csv         # Attendance records
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies list
```

## Troubleshooting
- **Face Detection Accuracy**: Adjust the similarity threshold in the `ensemble_similarity` function if the system struggles to differentiate similar faces.
- **Camera Access Issues**: Ensure webcam permissions are enabled for Streamlit.
- **Image Quality**: For best results, ensure good lighting when capturing faces, especially in live capture mode.

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Follow coding best practices and ensure that all code is tested before submission.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
