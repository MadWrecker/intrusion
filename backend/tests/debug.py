import os
import sqlite3
import json
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'factory_system.db')

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT employee_id, name, face_embedding, image_folder FROM Employees ORDER BY id DESC LIMIT 1")
row = cursor.fetchone()

if not row:
    print("No employees found.")
    exit()

emp_id = row['employee_id']
name = row['name']
emp_folder = row['image_folder']

print(f"Testing employee: {name} ({emp_id})")

if row['face_embedding']:
    saved_embs = json.loads(row['face_embedding'])
    saved_embs = [np.array(e) for e in saved_embs]
    print(f"Loaded {len(saved_embs)} embeddings from DB. Shape: {saved_embs[0].shape}")
else:
    print("No embeddings in DB.")
    exit()

# Find an image file
img_files = [f for f in os.listdir(emp_folder) if f.endswith('.jpg')]
if not img_files:
    print("No images found in folder:", emp_folder)
    exit()

img_path = os.path.join(emp_folder, img_files[0])
print("Using image:", img_path)

# Test 1: Full image extraction (like add_employee)
print("\n--- Test 1: Full Image ---")
try:
    reps_full = DeepFace.represent(img_path=img_path, model_name="ArcFace", detector_backend="opencv", enforce_detection=True, align=True)
    emb_full = np.array(reps_full[0]['embedding'])
    emb_full = emb_full / np.linalg.norm(emb_full)
    print("Full Image DB similarity:", np.dot(emb_full, saved_embs[0]))
except Exception as e:
    print("Full image failed:", e)

# Test 2: Cropped image extraction (like recognition.py)
print("\n--- Test 2: Haar Crop + Pad (Live Feed Simulation) ---")
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.2, 4)

if len(faces) > 0:
    x, y, w, fh = faces[0]
    rx1, ry1, rx2, ry2 = x, y, x + w, y + fh
    
    pad_x = int(w * 0.4)
    pad_y = int(fh * 0.4)
    h, w_img = img.shape[:2]
    
    px1, py1 = max(0, rx1 - pad_x), max(0, ry1 - pad_y)
    px2, py2 = min(w_img, rx2 + pad_x), min(h, ry2 + pad_y)
    
    face_crop = img[py1:py2, px1:px2].copy()
    
    cv2.imwrite("test_crop.jpg", face_crop)
    print("Saved test_crop.jpg. Size:", face_crop.shape)
    
    try:
        reps_crop = DeepFace.represent(img_path=face_crop, model_name="ArcFace", detector_backend="opencv", enforce_detection=True, align=True)
        emb_crop = np.array(reps_crop[0]['embedding'])
        emb_crop = emb_crop / np.linalg.norm(emb_crop)
        
        sim = np.dot(emb_crop, saved_embs[0])
        print("Live Feed Simulation similarity:", sim)
        if sim < 0.50:
            print("FAILED MATCH! Similarity < 0.50")
        else:
            print("SUCCESS! Match > 0.50")
    except Exception as e:
        print("Crop extraction failed:", e)
else:
    print("Haar cascade failed to find face in test image.")

conn.close()
