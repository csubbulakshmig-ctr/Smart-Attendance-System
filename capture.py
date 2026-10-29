import cv2
import os

# 🔥 D drive path
base_dir = r"D:\smart_attendance"
dataset = os.path.join(base_dir, "dataset")

# 👤 Enter name (folder must exist)
person_name = input("Enter name: ").lower().strip()
person_path = os.path.join(dataset, person_name)

if not os.path.exists(person_path):
    print("Folder not found ❌")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working ❌")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = len(os.listdir(person_path))

print("Press 'c' to capture image")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w > 60:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    # 🔥 PRESS C → capture one image
    if key == ord('c'):

        if len(faces) == 0:
            print("No face detected ❌")
            continue

        for (x, y, w, h) in faces:
            if w > 60:
                face = gray[y:y+h, x:x+w]

                if face.size == 0:
                    continue

                count += 1
                file_path = os.path.join(person_path, f"{count}.jpg")

                cv2.imwrite(file_path, face)

                print("Image saved ✅:", file_path)

                break  # 🔥 only one image

        break  # 🔥 exit after one capture

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
