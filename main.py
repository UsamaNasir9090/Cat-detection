import cv2
import os
import numpy as np

# Load the dataset of cat images
img_dir = 'D:\\cats'
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

# Create a new directory to store the cropped face images
cropped_dir = 'path/to/cropped/faces'
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

# Loop through each image in the dataset
for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If no face is detected, skip this image
    if len(faces) == 0:
        print(f"No face detected in {img_file}, skipping...")
        continue

    # Crop the face from the image
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]

    # Save the cropped face image
    cropped_img_path = os.path.join(cropped_dir, img_file)
    cv2.imwrite(cropped_img_path, face_img)

    print(f"Face detected and cropped in {img_file}, saved to {cropped_img_path}")

# Perform DBSCAN clustering
clt = DBSCAN(metric='precomputed', min_samples=3, eps=0.5)
clt.fit(distances)

# Get the cluster labels
labels = clt.labels_

# Create a dictionary to store the clustered faces
clustered_faces = {}
for i, label in enumerate(labels):
    if label not in clustered_faces:
        clustered_faces[label] = []
    clustered_faces[label].append(img_files[i])

# Print the clustered faces
for label, faces in clustered_faces.items():
    print(f"Cluster {label}: {', '.join(faces)}")