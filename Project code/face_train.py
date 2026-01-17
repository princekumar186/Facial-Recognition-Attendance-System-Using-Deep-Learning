import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
IMG_SIZE = (200, 200)

faces = []
labels = []
label_map = {}
label_id = 0

print("Loading dataset...")

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        faces.append(img)
        labels.append(label_id)

    label_id += 1

faces = np.array(faces)
labels = np.array(labels)

if len(faces) == 0:
    raise ValueError("❌ No training images found")

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)
model.save("face_model.xml")

np.save("labels.npy", label_map)

print("✅ Training completed successfully")
