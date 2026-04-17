import os
import cv2
import json
import numpy as np
from database import get_db_connection
from models.detector import FastFaceDetector
from models.recognizer import FastFaceRecognizer
from utils.augmentation import generate_augmentations

def upgrade_legacy_profiles():
    print("Starting Legacy Identity Upgrade to STRONG mode...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, employee_id, name, face_embedding, image_folder FROM Employees")
    rows = cursor.fetchall()
    
    detector = FastFaceDetector()
    recognizer = FastFaceRecognizer()
    
    upgraded_count = 0
    failed_count = 0
    
    for row in rows:
        emp_id = row['employee_id']
        name = row['name']
        emb_json = row['face_embedding']
        folder = row['image_folder']
        
        needs_upgrade = False
        if emb_json:
            embs = json.loads(emb_json)
            if len(embs) < 5:
                needs_upgrade = True
        else:
            needs_upgrade = True
            
        if not needs_upgrade:
            continue
            
        print(f"[{emp_id}] {name} is currently [WEAK]. Attempting upgrade...")
        
        # Look for a source image
        source_img_path = None
        if folder and os.path.exists(folder):
            for file in os.listdir(folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_img_path = os.path.join(folder, file)
                    break
                    
        if not source_img_path:
            print(f" -> ERROR: No source image found in folder '{folder}' for {name}. They must manually re-enroll to become STRONG.")
            failed_count += 1
            continue
            
        print(f" -> Found source image: {source_img_path}. Augmenting...")
        img = cv2.imread(source_img_path)
        if img is None:
            print(f" -> ERROR: Failed to read image {source_img_path}")
            failed_count += 1
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented_images = generate_augmentations(img_rgb)
        
        raw_embeddings = []
        for aug_img in augmented_images:
            try:
                boxes, landmarks_list = detector.detect(aug_img, return_landmarks=True)
                if not boxes: continue
                
                # Best box
                if len(boxes) > 1:
                    best_idx = 0
                    max_area = 0
                    for i, box in enumerate(boxes):
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        if area > max_area:
                            max_area = area
                            best_idx = i
                    x1, y1, x2, y2, _ = boxes[best_idx]
                    face_landmarks = landmarks_list[best_idx] if landmarks_list else None
                else:
                    x1, y1, x2, y2, _ = boxes[0]
                    face_landmarks = landmarks_list[0] if landmarks_list else None
                    
                h, w = aug_img.shape[:2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                face_img = aug_img[y1:y2, x1:x2]
                if face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
                    continue
                
                if face_landmarks is not None:
                    face_landmarks = [[float(lx - x1), float(ly - y1)] for (lx, ly) in face_landmarks]
                    
                emb = recognizer.get_embedding(face_img, face_landmarks)
                if emb is not None:
                    emb = np.array(emb, dtype=np.float32).flatten()
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                        raw_embeddings.append(emb.tolist())
            except Exception:
                pass
                
        if len(raw_embeddings) < 10:
            print(f" -> ERROR: Only generated {len(raw_embeddings)} valid augmentations. Source image quality too poor.")
            failed_count += 1
            continue
            
        # Clean Outliers
        raw_embeddings_np = np.array(raw_embeddings)
        centroid = np.mean(raw_embeddings_np, axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid)
        similarities = np.dot(raw_embeddings_np, centroid_norm)
        cleaned_embeddings = [raw_embeddings[i] for i, sim in enumerate(similarities) if sim > 0.65]
        
        if len(cleaned_embeddings) < 10:
            print(f" -> ERROR: Outlier removal resulted in only {len(cleaned_embeddings)} embeddings.")
            failed_count += 1
            continue
            
        # Success! Save back to database
        try:
            cursor.execute("UPDATE Employees SET face_embedding=? WHERE employee_id=?", 
                           (json.dumps(cleaned_embeddings), emp_id))
            conn.commit()
            print(f" -> SUCCESS: Upgraded {name} to [STRONG] with {len(cleaned_embeddings)} robust embeddings!")
            upgraded_count += 1
        except Exception as e:
            print(f" -> ERROR: DB Update failed: {e}")
            failed_count += 1

    conn.close()
    
    print("\n--- UPGRADE COMPLETE ---")
    print(f"Successfully upgraded to STRONG: {upgraded_count}")
    print(f"Failed (Require manual re-enroll via API): {failed_count}")

if __name__ == "__main__":
    upgrade_legacy_profiles()
