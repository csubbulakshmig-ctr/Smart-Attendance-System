import cv2
import os
import pickle
import csv
from datetime import datetime

# 🔥 PATH
base_dir = r"D:\smart_attendance"
dataset = os.path.join(base_dir, "dataset")

# 🔥 LOAD MODEL
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"D:\smart_attendance\trainer.yml")

with open(r"D:\smart_attendance\names.pkl", "rb") as f:
    names = pickle.load(f)

# 🔥 FACE DETECTOR
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 🔥 ATTENDANCE FILE
today = datetime.now().strftime("%Y-%m-%d")
attendance_file = os.path.join(base_dir, "attendance.csv")

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

marked = set()
with open(attendance_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 1 and row[1] == today:
            marked.add(row[0])

cap = cv2.VideoCapture(0)

print("System started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        if w > 60:
            face = gray[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (200, 200))

            id, conf = recognizer.predict(face)

            # 🔥 STRICT CONDITION
            if conf < 60:
                name = names.get(id, "Unknown")

                print(f"👉 {name} face → {name} ✅")

                # 🔥 Attendance
                if name not in marked:
                    with open(attendance_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            name,
                            today,
                            datetime.now().strftime("%H:%M:%S")
                        ])
                    marked.add(name)

                    # ✅ CLEAN OUTPUT
                    print(f"{name} attendance marked ✅")
                else:
                    print(f"{name} already marked")

                # ✅ ONLY GREEN BOX FOR KNOWN
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, name,
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)

            else:
                # ❌ UNKNOWN → NOTHING (no box, no text)
                pass

    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
