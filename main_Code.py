import cv2
import numpy as np
import face_recognition
import os
import pyttsx3
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
from pynput.mouse import Controller, Button
import HandTrackingModule as htm
import time

# Paths
face_folder = 'faces'
os.makedirs(face_folder, exist_ok=True)
attendance_file = 'Attendance.csv'

# Voice Engine
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Encode known faces
def encode_faces():
    images = []
    classNames = []
    myList = os.listdir(face_folder)
    for cl in myList:
        curImg = cv2.imread(f'{face_folder}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])

    encodeList = []
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print(f"Skipping {classNames[i]} - No face found.")
            continue
    return encodeList, classNames

# Mark attendance
def markAttendance(name):
    with open(attendance_file, 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Register New Face with Cropping
def register_new_face():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    speak("Please look at the camera to capture your face.")
    time.sleep(2)
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_crop = frame[top:bottom, left:right]

            name = simpledialog.askstring("Register", "Enter your name:")
            if name:
                cv2.imwrite(f'{face_folder}/{name}.jpg', face_crop)
                speak("Registration successful. Please restart the system.")
            else:
                speak("Registration cancelled.")
        else:
            speak("No face detected. Please try again.")
            messagebox.showerror("Error", "No face detected. Try again.")
    else:
        speak("Failed to capture image. Try again.")
    cap.release()
    cv2.destroyAllWindows()

# Face Recognition Validation
def face_recognition_validation():
    encodeListKnown, classNames = encode_faces()
    if not encodeListKnown:
        speak("No faces registered. Please register first.")
        messagebox.showerror("Error", "No faces registered.")
        return False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    authenticated = False
    speak("Face recognition started. Look at the camera.")

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            if len(faceDis) == 0:
                continue
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex]
                markAttendance(name)
                speak(f"Welcome {name}")
                authenticated = True
                cap.release()
                cv2.destroyAllWindows()
                return True

        cv2.imshow('Face Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if not authenticated:
        cap.release()
        cv2.destroyAllWindows()
        speak("Face not recognized. Please register.")
        messagebox.showerror("Unauthorized", "Face not recognized. Please register first.")
        return False

# Virtual Mouse Control
def start_virtual_mouse():
    wCam, hCam = 640, 480
    frameR = 100
    smoothening = 7
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.handDetector(maxHands=1)
    mouse = Controller()
    screen_width, screen_height = 1920, 1080

    speak("Virtual Mouse Activated")

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (screen_width, 0))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, screen_height))

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                mouse.position = (clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    mouse.click(Button.left, 1)
                    time.sleep(0.2)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime != 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Virtual Mouse", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Window
def gui_window():
    root = tk.Tk()
    root.title("Face Recognition Virtual Mouse")
    root.geometry("400x250")

    tk.Label(root, text="Face Recognition Virtual Mouse", font=("Arial", 16)).pack(pady=20)

    def login():
        root.destroy()
        result = face_recognition_validation()
        if result:
            start_virtual_mouse()

    def register():
        register_new_face()

    tk.Button(root, text="Login", command=login, width=20, height=2).pack(pady=10)
    tk.Button(root, text="Register New Face", command=register, width=20, height=2).pack(pady=10)
    tk.Button(root, text="Exit", command=root.quit, width=20, height=2).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    gui_window()
