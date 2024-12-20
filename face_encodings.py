import face_recognition
import os
import pickle
import cv2

def save_known_faces(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
    with open(filename, 'rb') as f:
        datc=pickle.load(f)
        with open("master_file_j.pkl", 'wb') as mf:
            pickle.dump(datc,mf)
        print(datc)
def encode_images_from_folder(folder_path, encoding_file):
    all_photos=os.listdir(folder_path)
    data={}
    for i in all_photos:
        print(i)
        image_path=rf'C:\Users\mvasu\PycharmProjects\FRASbiz\dataii\{i}'
        image=cv2.imread(image_path)
        tempdic={}
        #image=face_recognition.load_image_file(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(gray)
        if face_locations:
            face_encoding = face_recognition.face_encodings(image, face_locations)
            tempdic['face_encodings']=face_encoding
        if '.JPG' in i:
            data[i.replace('.JPG', '')] = tempdic
        else:
            data[i.replace('.jpg', '')] = tempdic
    print(data)

    save_known_faces(data, encoding_file)

def main(folder_path, encoding_file):
    encode_images_from_folder(folder_path, encoding_file)
    print("Encoding completed!")

# Specify folder path and encoding file here
folder_path = r"dataii"
encoding_file = "known_faces_j.pkl"

main(folder_path, encoding_file)
