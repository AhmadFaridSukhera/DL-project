# train.py
import os

import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Initialize face detector and feature extractor
detector = MTCNN()
embedder = FaceNet()

# Path to the folder containing subfolders with known faces
known_faces_dir = '12ab/pics'
known_faces = {}

# Load known faces and compute their embeddings
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        image = cv2.imread(filepath)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)

        for face in faces:
            x, y, width, height = face['box']
            face_crop = image_rgb[y:y+height, x:x+width]

            # Check if the crop contains enough pixels
            if face_crop.size == 0:
                continue

            face_features = embedder.embeddings([face_crop])[0]
            known_faces[person_name] = known_faces.get(person_name, [])
            known_faces[person_name].append(face_features)

# Compute the mean of feature vectors for each person
mean_known_faces = {name: np.mean(features, axis=0) for name, features in known_faces.items()}

# Save known faces data to a file
with open('known_faces.pkl', 'wb') as f:
    pickle.dump(mean_known_faces, f)

print("Known faces data saved to 'known_faces.pkl'")