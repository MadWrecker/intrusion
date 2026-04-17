import os
import cv2
import numpy as np
import json
import sqlite3
import datetime
from database import get_db_connection
from models.detector import FastFaceDetector
from models.recognizer import FastFaceRecognizer
from utils.augmentation import generate_augmentations

def register_master_employees(master_folder="../master_employees"):
    print(f"\n[Master Registration] Starting bulk enrollment from: {master_folder}")
    
    if not os.path.exists(master_folder):
        print(f"[ERROR] The folder '{master_folder}' does not exist. Please create it.")
        print("Folder structure should be:\nmaster_employees/\n    ├── John Doe/\n    │   └── face1.jpg\n    ├── Jane Smith/\n    │   └── profile.png")
        os.makedirs(master_folder, exist_ok=True)
        return

    detector = FastFaceDetector()
    recognizer = FastFaceRecognizer()
    
    conn = get_db_connection()
    cursor = conn.cursor()

    # Iterate over every sub-directory in master_employees (which represents the employee's name)
    for employee_name in os.listdir(master_folder):
        employee_path = os.path.join(master_folder, employee_name)
        
        if not os.path.isdir(employee_path):
            continue
            
        print(f"\n--- Processing Employee: {employee_name} ---")
        
        # We will use the name as both the name and a default ID/department
        employee_id = employee_name.lower().replace(" ", "_")
        department = "General"
        phone = ""
        
        all_embeddings = []
        images_processed = 0
        
        for image_name in os.listdir(employee_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            image_path = os.path.join(employee_path, image_name)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"  [X] Failed to read {image_name}")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented_images = generate_augmentations(img_rgb)
            
            raw_embeddings = []
            for aug_img in augmented_images:
                try:
                    boxes, landmarks_list = detector.detect(aug_img, return_landmarks=True)
                    if not boxes: continue
                        
                    # Pick LARGEST face
                    best_idx = 0
                    if len(boxes) > 1:
                        max_area = 0
                        for i, box in enumerate(boxes):
                            area = (box[2] - box[0]) * (box[3] - box[1])
                            if area > max_area:
                                max_area = area
                                best_idx = i
                    
                    x1, y1, x2, y2, score = boxes[best_idx]
                    if score < 0.75: continue
                    
                    # 20% Crop Padding Math (MIRRORED EXACTLY FROM LIVE PIPELINE)
                    h, w = aug_img.shape[:2]
                    pad_x = int((x2 - x1) * 0.2)
                    pad_y = int((y2 - y1) * 0.2)
                    px1 = max(0, int(x1) - pad_x)
                    py1 = max(0, int(y1) - pad_y)
                    px2 = min(w, int(x2) + pad_x)
                    py2 = min(h, int(y2) + pad_y)
                    
                    face_img = aug_img[py1:py2, px1:px2].copy()
                    if face_img.size == 0 or face_img.shape[0] < 25 or face_img.shape[1] < 25:
                        continue
                        
                    # SR & Enhancements (MIRRORED)
                    try:
                        if w < 60 and h < 60:
                            if not hasattr(detector, 'sr_model'):
                                model_path = os.path.join(os.path.dirname(__file__), 'models', 'weights', 'LapSRN_x4.pb')
                                if os.path.exists(model_path):
                                    from cv2 import dnn_superres
                                    detector.sr_model = dnn_superres.DnnSuperResImpl_create()
                                    detector.sr_model.readModel(model_path)
                                    detector.sr_model.setModel("lapsrn", 4)
                            if getattr(detector, 'sr_model', None):
                                face_img = detector.sr_model.upsample(face_img)

                        l, a, b = cv2.split(cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB))
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                        l_cl = clahe.apply(l)
                        face_enhanced = cv2.cvtColor(cv2.merge((l_cl, a, b)), cv2.COLOR_LAB2BGR)
                        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
                        face_img = cv2.filter2D(face_enhanced, -1, kernel)
                    except Exception:
                        pass
                        
                    # YuNet Re-Scale (MIRRORED)
                    try:
                        face_img_large = cv2.resize(face_img, (160, 160))
                        boxes_large, lms_large = detector.detect(face_img_large, return_landmarks=True)
                        face_landmarks = None
                        if lms_large and boxes_large:
                            center_x, center_y = 80, 80
                            best_dist = float('inf')
                            for c_box, c_lm in zip(boxes_large, lms_large):
                                bx1_c, by1_c, bx2_c, by2_c = c_box[:4]
                                bcx, bcy = (bx1_c + bx2_c) // 2, (by1_c + by2_c) // 2
                                dist = (bcx - center_x)**2 + (bcy - center_y)**2
                                if dist < best_dist:
                                    best_dist = dist
                                    scale_x, scale_y = face_img.shape[1] / 160.0, face_img.shape[0] / 160.0
                                    lm_scaled = c_lm.copy()
                                    lm_scaled[:, 0] *= scale_x
                                    lm_scaled[:, 1] *= scale_y
                                    face_landmarks = lm_scaled
                    except Exception:
                        face_landmarks = None

                    emb = recognizer.get_embedding(face_img, face_landmarks)
                    if emb is not None:
                        emb = np.array(emb, dtype=np.float32).flatten()
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                            raw_embeddings.append(emb.tolist())
                except Exception as e:
                    pass
            
            # Outlier Cleanup Math
            if len(raw_embeddings) >= 5:
                raw_embeddings_np = np.array(raw_embeddings)
                centroid = np.mean(raw_embeddings_np, axis=0)
                centroid_norm = centroid / np.linalg.norm(centroid)
                similarities = np.dot(raw_embeddings_np, centroid_norm)
                cleaned_embeddings = [raw_embeddings[i] for i, sim in enumerate(similarities) if sim > 0.65]
                
                if len(cleaned_embeddings) >= 5:
                    all_embeddings.extend(cleaned_embeddings)
                    images_processed += 1
                    print(f"  [+] Mapped {len(cleaned_embeddings)} vectors from {image_name}")
                else:
                    print(f"  [!] {image_name} failed outlier consistency test.")
            else:
                print(f"  [!] {image_name} failed baseline quality filter.")

        if all_embeddings:
            embeddings_json = json.dumps(all_embeddings)
            try:
                # Remove existing employee with same ID before replacing him
                cursor.execute("DELETE FROM Employees WHERE employee_id=?", (employee_id,))
                
                cursor.execute("INSERT INTO Employees (employee_id, name, department, phone, face_embedding, image_folder) VALUES (?, ?, ?, ?, ?, ?)", 
                               (employee_id, employee_name, department, phone, embeddings_json, ""))
                conn.commit()
                print(f"[OK] Successfully registered '{employee_name}' with {len(all_embeddings)} secure identity vectors.")
            except Exception as e:
                print(f"[ERROR] Database error for {employee_name}: {e}")
        else:
            print(f"[FAIL] Could not register '{employee_name}'. Ensure high quality face images are provided.")
    
    conn.close()
    
    print("\n[Master Registration] Done! Please restart your camera feed to sync new identities.")
    try:
        from camera import engine
        if engine:
            engine.load_embeddings_from_db()
            print("[Master Registration] Memory updated automatically.")
    except Exception:
        pass

if __name__ == "__main__":
    register_master_employees()
