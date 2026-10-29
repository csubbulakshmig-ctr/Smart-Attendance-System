import cv2
import os
import numpy as np
import pickle

# 🔥 PATH (D DRIVE)
dataset = r"D:\smart_attendance\dataset"

faces = []
labels = []
names = {}

label_id = 0

print("Training started...")

for person in os.listdir(dataset):

    person_path = os.path.join(dataset, person)

    if not os.path.isdir(person_path):
        continue

    names[label_id] = person
    print("Label:", label_id, "Name:", person)

    for img in os.listdir(person_path):

        img_path = os.path.join(person_path, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        # 🔥 Resize for better accuracy
        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(label_id)

    label_id += 1

# 🔥 Check if data exists
if len(faces) == 0:
    print("No images found ❌")
    exit()

# 🔥 TRAIN MODEL
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# 🔥 SAVE FILES (IMPORTANT)
recognizer.save(r"D:\smart_attendance\trainer.yml")

with open(r"D:\smart_attendance\names.pkl", "wb") as f:
    pickle.dump(names, f)

print("Training completed ✅")
