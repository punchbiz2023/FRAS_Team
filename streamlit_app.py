import base64
import os
from dotenv import load_dotenv,dotenv_values
import json
import streamlit as st
from streamlit import session_state
from streamlit_navigation_bar import st_navbar
import E_Upload
import F_capture
import gspread
from oauth2client.service_account import ServiceAccountCredentials

load_dotenv()
# Construct JSON using environment variables
credentials = {
    "type": "service_account",
    "project_id": "attendance-system-439310",
    "private_key_id": os.getenv("PRIVATE_KEY_ID"),
    "private_key": os.getenv("PRIVATE_KEY").replace("\\n", "\n"),  # Correct newline format for private key
    "client_email": os.getenv("CLIENT_EMAIL"),
    "client_id": os.getenv("CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
    "universe_domain": "googleapis.com"
}


# Initialize session state variables for authentication and navigation
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'facultyname' not in st.session_state:
    st.session_state.facultyname = ""
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False

# Set Streamlit configuration
st.set_page_config(page_title="FRASbiz", page_icon="♾️",
                   initial_sidebar_state="collapsed")  # Add custom CSS for background color and header bar
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');


    *{
      font-family: "Inter", sans-serif !important;
      }
    /* Set background color of the page to black */
    body, .stApp {
        background-color: #0e1117;
    }

    /* Customize text color for visibility */
    .stApp, h1, h2, h3, h4, h5, h6, p, div {
        color: white;
    }

    /* Customize Streamlit's default header bar */
    header {
        background-color: #0E1117 !important;
        color: white;
    }

    /* Additional customizations */
    .css-1aumxhk {
        background-color: #0E1117 !important;
        color: white;
    }

    /* Customize sidebar to match the theme */
    .css-18e3th9 {
        background-color: #0E1117 !important;
    }


    /* Button specific styling */
    .stButton>button {
        background-color: transparent; 
        color: dodgerblue;
        border:solid 2px dodgerblue;
        font-weight: bold;
        padding: 10px 40px;
        transition : all .3s ease;
    }
    .stButton>button:hover {
       background-color: dodgerblue; 
        color: #ffffff;
        scale: 1.1;
        border:solid 2px dodgerblue;
    }
    .stDownloadbutton>button {
        background-color: transparent; 
        color: dodgerblue;
        border:solid 2px dodgerblue;
        font-weight: bold;
        padding: 10px 40px;
        transition : all .3s ease;
    }
    .stDownloadbutton>button:hover {
       background-color: dodgerblue; 
        color: #ffffff;
        scale: 1.1;
        border:solid 2px dodgerblue;
    }

    /* Add a top margin to move the navigation bar down */
    .custom-navbar {
        margin-top: 50px; /* Adjust this value to control the navbar position */
    }
    </style>
    """,
    unsafe_allow_html=True
)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials, scope)
client = gspread.authorize(creds)
sheet = client.open("AttendanceSystemUsers").sheet1  # Open the Google Sheet

attendance_files = {

    "II-AI & DS": r"Attendance_j.csv",  # Replace with actual file path
    "III-AI & DS": r"Attendance_s.csv"
}# Replace with actual file path

# Define pages and styling for navigation bar
pages = ["Home", "Upload Image", "Capture Image","LogOut"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path_1 = r"kot.svg"
logo_path_2 = r"punch.svg"
logo_path=r"logo.svg"
styles = {
    "nav": {
        "box-sizing": "border-box",
        "background-color": "#ffffff",
        "justify-content": "left",
        "border-radius": "31px",
        "height": "50px",

        "width": "100%",
        "padding-block": "8px",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "#000000",
        "padding": "14px",
        "transition": "all 0.3s ease",

        "border-radius": "31px",
    },
    "active": {
        "background-color": "dodgerblue",
        "color": "#ffffff",
        "font-weight": "normal",
        "padding": "14px",
        "margin-block": "0.0125rem",
        "transition": "all 0.3s ease",

    }
}
options = {
    "show_menu": False,
    "show_sidebar": False,
}
# Navigation bar selection logic after successful login
if st.session_state.is_authenticated:
    st.markdown('<div class="custom-navbar">', unsafe_allow_html=True)
    st.markdown("""
        <style>
            .bottom-logo-container {
                position: fixed;
                bottom: 10px;  /* Adjust the space from the bottom */
                left: 0;
                right: 0;
                text-align: center;
                background-color: white;
                padding: 10px;
                display: flex;
                justify-content: center;
                gap: 20px;
            }
            .bottom-logo-container img {
                height: 40px;  /* Adjust the logo size */
            }
            .bottom-logo-container .label {
                display: block;
                font-size: 12px;
                margin-top: 5px;
                color: #333;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display the logos at the bottom
    st.markdown("""
        <div class="bottom-logo-container">
            <span class="label">Associated with</span>
        </div>
        <div class="bottom-logo-container">
            <img src="data:image/svg+xml;base64,{}" />
            <img src="data:image/svg+xml;base64,{}" />
        </div>
        
    """.format(
        base64.b64encode(open(logo_path_1, 'rb').read()).decode(),
        base64.b64encode(open(logo_path_2, 'rb').read()).decode()
    ), unsafe_allow_html=True)
    page = st_navbar(
        pages,
        logo_path=logo_path,
        styles=styles,
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    page = None


# Function to check if a username exists in the Google Sheet
def check_user_in_sheet(username):
    users = sheet.get_all_records()
    for user in users:
        if user['username'] == username:
            return True
    return False


# Function to check if the password matches the provided username
def check_password_in_sheet(username, password):
    users = sheet.get_all_records()
    for user in users:
        if user['username'] == username:
            return user['password'] == password
    return False


# Function to retrieve faculty name for a user if credentials match
def get_user_details(username, password):
    users = sheet.get_all_records()
    for user in users:
        if user['username'] == username and user['password'] == password:
            return user['facultyname']
    return None


# Function to add a new user to the Google Sheet
def add_user_to_sheet(facultyname, username, email, password):
    sheet.append_row([facultyname, username, email, password])
    st.write(f"User {facultyname} added successfully!")


# Placeholder functions for page navigation
def go_to_signup():
    st.session_state.page = 'signup'


def go_to_login():
    st.session_state.page = 'login'


# Function to handle the login
def handle_login():
    login_username = st.session_state.login_username
    login_password = st.session_state.login_password

    if check_user_in_sheet(login_username):
        if check_password_in_sheet(login_username, login_password):
            st.session_state.facultyname = get_user_details(login_username, login_password)
            st.session_state.is_authenticated = True
            st.session_state.page = 'home'
            st.success("Logged in successfully!")
        else:
            st.error("Incorrect password. Please try again.")
    else:
        st.error("Username doesn't exist.")


# Login page layout
def login_page():
    st.title("Face Recognition and Attendance System")
    st.subheader("Welcome! Please Sign In")
    st.text_input("Username", key="login_username")
    st.text_input("Password", type='password', key="login_password")
    col1, col_spacer, col2 = st.columns([2, 5, 2.2])

    # Place the Login button in the first column and Sign Up button in the last column
    with col1:
        st.button("Login", on_click=handle_login)
    with col2:
        st.button("Sign-Up", on_click=go_to_signup)


# Sign-up page layout
def signup_page():
    st.title('Face Recognition and Attendance System')
    st.subheader("Sign Up")
    facultyname = st.text_input("Faculty Name", key='signup_facultyname')
    username = st.text_input("Username", key="signup_username")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type='password', key="signup_password")
    confirmpassword = st.text_input("Confirm Password", type='password', key="confirm_password")

    if st.button("Sign Up"):
        if facultyname and username and email and password and confirmpassword:
            if check_user_in_sheet(username):
                st.error("Username already exists. Please choose a different username.")
            elif password != confirmpassword:
                st.error("Passwords do not match. Please re-enter the passwords.")
            else:
                add_user_to_sheet(facultyname, username, email, password)
                st.success("Sign-up successful! Please proceed to the Log In page.")
        else:
            st.error("Please fill in all fields.")
    st.button("Go to Login", on_click=go_to_login)


# Class Selection Page layout

# Image Upload Page
def upload():
    E_Upload.upload_image_page(st.session_state.known_faces_file, st.session_state.attendance_file)


# Image Capture Page
def capture():
    F_capture.live_capture_page(st.session_state.known_faces_file, st.session_state.attendance_file)


# Home page layout with attendance download option
def home_page():
    st.title("FRASbiz")

    st.subheader(f"Welcome, {st.session_state.facultyname}!")

    # Class selection dropdown for attendance download
    class_selected = st.selectbox("Select Class for Attendance", list(attendance_files.keys()))
    st.write(f"Selected Class: {class_selected}")
    if class_selected == "III-AI & DS":
        st.session_state.known_faces_file = 'known_faces_s.pkl'
        st.session_state.master_file = 'master_file_s.pkl'
        st.session_state.attendance_file = 'Attendance_s.csv'
    else:
        st.session_state.known_faces_file = 'known_faces_j.pkl'
        st.session_state.master_file = 'master_file_j.pkl'
        st.session_state.attendance_file = 'Attendance_j.csv'

    # Get the file path for the selected class
    attendance_file_path = attendance_files[class_selected]

    # Read the attendance file for download
    with open(attendance_file_path, "rb") as file:
        attendance_data = file.read()

    # Display the download button
    st.download_button(
        label="Download Attendance",
        data=attendance_data,
        file_name=f"{class_selected}_attendance.csv",
        mime="text/csv"
    )
    def logout():
        st.session_state.is_authenticated = False
        st.session_state.page = 'login'


    if st.button("Logout", key="logout_button", on_click=logout):
        st.experimental_rerun()


# Main app logic
if st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'signup':
    signup_page()
elif st.session_state.is_authenticated:
    if page == "Home":
        home_page()
    # elif page == "Class Selection":
    # lass_select()
    elif page == "Upload Image":
        upload()
    elif page == "Capture Image":
        capture()
    elif page=="LogOut":
        st.session_state.is_authenticated = False
        if not st.session_state.is_authenticated:
            login_page()
