import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import os
import winsound

IMG_SIZE = (200, 200)

# -------- AUDIO FILES --------
AUDIO_DIR = "audio"
SUCCESS_SOUND = os.path.join(AUDIO_DIR, "success.wav")
UNKNOWN_SOUND = os.path.join(AUDIO_DIR, "unknown.wav")
WELCOME_SOUND = os.path.join(AUDIO_DIR, "welcome.wav")

# Load model & labels
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.xml")
label_map = np.load("labels.npy", allow_pickle=True).item()

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Excel attendance file
ATT_FILE = "attendance.xlsx"
if not os.path.exists(ATT_FILE):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel(ATT_FILE, index=False)

marked_today = set()
today = datetime.now().strftime("%Y-%m-%d")

# Auto camera detection
cap = None
for i in range(5):
    temp = cv2.VideoCapture(i)
    if temp.isOpened():
        cap = temp
        print(f"Using camera index {i}")
        break

if cap is None:
    print("❌ No camera found")
    exit()

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y+10:y+h-10, x+10:x+w-10]
        roi = cv2.resize(roi, IMG_SIZE)

        label, confidence = model.predict(roi)

        if confidence < 150:
            name = label_map[label]

            if name not in marked_today:
                time_now = datetime.now().strftime("%H:%M:%S")

                df = pd.read_excel(ATT_FILE)
                df.loc[len(df)] = [name, today, time_now]
                df.to_excel(ATT_FILE, index=False)

                marked_today.add(name)
                print(f"✔ Attendance marked for {name}")

                if os.path.exists(WELCOME_SOUND):
                    winsound.PlaySound(WELCOME_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC)

                if os.path.exists(SUCCESS_SOUND):
                    winsound.PlaySound(SUCCESS_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC)

        else:
            name = "Unknown"
            if os.path.exists(UNKNOWN_SOUND):
                winsound.PlaySound(UNKNOWN_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC)

        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)

        cv2.putText(frame, f"{name} ({int(confidence)})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
