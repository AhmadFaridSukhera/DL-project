from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import cv2  
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

app = Flask(__name__)
CORS(app)

# Load known faces data from the saved file
try:
    with open('known_faces.pkl', 'rb') as f:
        mean_known_faces = pickle.load(f)
except FileNotFoundError:
    mean_known_faces = {}

# Initialize face detector and feature extractor
detector = MTCNN()
embedder = FaceNet()

# Define the similarity threshold for face recognition
THRESHOLD = 1.0

def recognize_faces(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Unable to open image {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        recognized_names = []

        for face in faces:
            x, y, width, height = face['box']
            face_crop = image_rgb[y:y+height, x:x+width]
            if face_crop.size == 0:
                continue

            face_features = embedder.embeddings([face_crop])[0]
            min_distance = float('inf')
            recognized_name = 'Unknown'

            for name, known_features in mean_known_faces.items():
                distance = np.linalg.norm(face_features - known_features)
                if distance < min_distance:
                    min_distance = distance
                    recognized_name = name

            if min_distance <= THRESHOLD:
                recognized_names.append(recognized_name)
            else:
                recognized_names.append('Unknown')

        return recognized_names
    except Exception as e:
        print(f"Error processing image: {e}")
        return []

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400    

        file = request.files['image']
        image_path = os.path.join('/tmp', file.filename)
        file.save(image_path)

        recognized_faces = recognize_faces(image_path)
        if not recognized_faces:
            raise ValueError('Face recognition failed or no faces found')

        return jsonify({'recognized_faces': recognized_faces})

    except Exception as e:
        print(f"Error in /recognize endpoint: {e}")
        return jsonify({'error': f"Error in /recognize endpoint: {e}"}), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

