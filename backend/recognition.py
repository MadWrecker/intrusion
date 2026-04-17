import cv2
import threading
import os
import os
# Force GPU visibility globally for all neural networks
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info(f"[DEBUG] GPU Hardware Accelerated for Recognition: {physical_devices[0]}")
    else:
        logger.info("[DEBUG] No GPU detected. Falling back to CPU for Recognition.")
except Exception:
    pass

try:
    import torch
    if torch.cuda.is_available():
        logger.info(f"Running on GPU. Hardware Accelerated for Detection/Recognition: {torch.cuda.get_device_name(0)}")
except Exception:
    pass

import time
import time
import numpy as np
import json
import queue
from utils.logger import get_logger

logger = get_logger('recognition')
from models.detector import FastFaceDetector
from models.recognizer import FastFaceRecognizer
from faiss_db import EmployeeVectorDB
from database import get_db_connection
from alerts import trigger_intruder_alert
from attendance_manager import mark_attendance
from tracker import SORTTracker

class RecognitionEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        
        # High quality SORT tracker
        self.tracker = SORTTracker(max_disappeared=15)
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(self.base_dir, 'employees')
        os.makedirs(self.db_path, exist_ok=True)
        
        self.track_states = {}
        self.track_labels = {}
        self.identity_history = {} # Stores rolling queue of matches per object_id
        
        # Frame control states for asynchronous multi-threaded rendering
        self.current_rects = []
        self.current_objects = {}
        
        # Multi-frame verification: dict mapping ID -> list of positive matched employee_ids
        self.track_verifications = {}
        self.VERIFICATION_FRAMES_REQUIRED = 3
        
        # Thread 3 Recognition Queue (Max 10 items to prevent GPU multi-second backlog lag)
        self.recognition_queue = queue.Queue(maxsize=10)
        
        # Load and cache employee embeddings (Memory caching & NumPy indexing)
        # Load and cache employee embeddings (Memory caching & NumPy indexing)
        self.known_names = {}
        
        logger.info("[DEBUG] Loading High-Performance GPU Models...")
        self.detector = FastFaceDetector()
        self.align_detector = FastFaceDetector()
        self.recognizer = FastFaceRecognizer()
        
        # Initialize FAISS Vector Database
        self.faiss_db = EmployeeVectorDB()
        self.load_embeddings_from_db()
            
        # 5-minute Cooldown Cache { "emp_id": timestamp, "intruder": timestamp }
        self.cooldown_cache = {}
        self.COOLDOWN_SECONDS = 300
        self.alert_cooldowns = {} # For attendance marking
        
        # System-Wide Diversity Auto-Reject Variables
        self.safe_mode_until = 0
        self.recent_assigned_identities = [] # Tracks (object_id, assigned_name, timestamp)

    def is_blurry(self, img, threshold=50.0):
        # cv2.Laplacian to calculate variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    # Removed strict brightness threshold so normal faces aren't rejected

    def load_embeddings_from_db(self):
        logger.info("[DEBUG] Loading embeddings into FAISS memory as Centroids...")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT employee_id, name, face_embedding FROM Employees")
        rows = cursor.fetchall()
        self.faiss_db.index.reset()
        self.faiss_db.employee_names = []
        self.faiss_db.employee_strengths = []
        for r in rows:
            emp_id = r['employee_id']
            self.known_names[emp_id] = r['name']
            if r['face_embedding']:
                try:
                    emb_list = json.loads(r['face_embedding'])
                    if not emb_list:
                        continue
                        
                    # STRONG vs WEAK check
                    if len(emb_list) >= 5:
                        strength = "STRONG"
                    else:
                        strength = "WEAK"
                        
                    embeddings = np.array(emb_list, dtype=np.float32)
                    
                    for emb in embeddings:
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                            
                        self.faiss_db.index.add(np.array([emb], dtype=np.float32))
                        self.faiss_db.employee_names.append(emp_id)
                        self.faiss_db.employee_strengths.append(strength)
                        
                    logger.info(f"[DEBUG] Loaded {len(embeddings)} independent vectors for {emp_id} [{strength}]")
                except Exception as e:
                    logger.info(f"Error parsing embedding for {emp_id}: {e}")
        conn.close()
        logger.info(f"[DEBUG] FAISS Loaded {self.index.ntotal if hasattr(self, 'index') else self.faiss_db.index.ntotal} specific Centers into fast search.")

    def start(self):
        self.running = True
        
        # Thread 3 -> recognition
        t_rec = threading.Thread(target=self.recognition_process_thread)
        t_rec.daemon = True
        t_rec.start()

    def get_embedding(self, img):
        # Helper for main.py (not used dynamically anymore, but kept for interface consistency if needed)
        pass

    def _is_in_detection_zone(self, box, frame_w, frame_h):
        # High Accuracy Detection Zone Rule
        # detection zone: W*0.2 to W*0.8, H*0.3 to H*0.9
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        z_x1 = frame_w * 0.2
        z_x2 = frame_w * 0.8
        z_y1 = frame_h * 0.3
        z_y2 = frame_h * 0.9
        
        return (z_x1 <= cx <= z_x2) and (z_y1 <= cy <= z_y2)

    def draw_annotations(self, out_frame):
        with self.lock:
            # We must use the precise tracker box assignments, not a heuristic nearest-neighbor map, to prevent overlapping ghost boxes.
            tracked_rects = getattr(self, 'current_tracked_rects', {})
            disappeared_counts = getattr(self, 'current_disappeared', {})
            
            for object_id, rect in tracked_rects.items():
                if disappeared_counts.get(object_id, 0) > 0:
                    continue # Do not draw bounding boxes for lost objects that are just coasting in tracker memory!
                    
                rx1, ry1, rx2, ry2 = [int(v) for v in rect]
                    
                state = self.track_states.get(object_id, "TRACKING")
                
                # Fetch tracked identity history, default to Unknown
                raw_label = self.track_labels.get(object_id, "Unknown")
                
                # Format output consistently to strict requirements
                if raw_label not in ["Unknown", "INTRUDER ALERT"]:
                    label = f"Known: {raw_label}"
                else:
                    label = "Unknown"
                
                # Unknown is Red, Identified is Green
                color = (0, 255, 0) if label.startswith("Known") else (0, 0, 255)
                
                cv2.rectangle(out_frame, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.putText(out_frame, label, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if state in ["TRUSTED", "INTRUDER"]:
                    cv2.rectangle(out_frame, (rx1, ry1-35), (rx1 + len(label)*10, ry1), color, -1)
                    cv2.putText(out_frame, label, (rx1+2, ry1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if state=="TRUSTED" else (255,255,255), 2)

    def process_frame(self, frame):
        if not self.running:
            return
            
        h, w = frame.shape[:2]
        
        # Removed ROI restriction to scan the entire frame
        z_x1, z_y1 = 0, 0
        roi = frame.copy()
        
        rects = []
        try:
            res = self.detector.detect(roi, return_landmarks=True)
            boxes, lms_list = res if isinstance(res, tuple) and len(res) == 2 else (res, [])
            for i, box in enumerate(boxes):
                # Assuming detector returns [x1, y1, x2, y2, score, ...]
                rx1, ry1, rx2, ry2 = [int(v) for v in box[:4]]
                rw = rx2 - rx1
                rh = ry2 - ry1
                conf = box[4] if len(box) > 4 else 1.0
                
                # Strict constraints to ignore non-human objects and false positives (like phones or background paintings)
                if rw < 40 or rh < 40:
                    continue
                if conf < 0.70:
                    continue
                
                # Aspect Ratio Validation: Human faces are generally squares or slightly tall rectangles.
                # If a box is extremely wide or extremely tall, it is a hardware hallucination (wall edge, shadow, etc).
                shape_ratio = rh / rw
                if shape_ratio < 0.8 or shape_ratio > 1.8:
                    continue
                    
                # Reject detections without clear facial features (eyes, nose, mouth)
                if i < len(lms_list):
                    lms = lms_list[i]
                    valid_features = sum(1 for pt in lms if pt[0] > 0 and pt[1] > 0)
                    if valid_features < 3: # Need at least 3 distinct valid facial features
                        continue
                        
                rects.append((rx1 + z_x1, ry1 + z_y1, rx2 + z_x1, ry2 + z_y1))
        except Exception as e:
            logger.info(f"[Detector Error] {e}")

        with self.lock:
            objects = self.tracker.update(rects)
            self.current_rects = rects.copy()
            self.current_objects = objects.copy()
            self.current_tracked_rects = self.tracker.object_rects.copy()
            self.current_disappeared = self.tracker.disappeared.copy()
            
            # MEMORY LEAK FIX: Clean up tracking states for permanently deregistered objects
            active_ids = set(self.tracker.objects.keys()).union(set(self.tracker.disappeared.keys()))
            for old_id in list(self.track_states.keys()):
                if old_id not in active_ids:
                    self.track_states.pop(old_id, None)
                    self.track_labels.pop(old_id, None)
                    self.identity_history.pop(old_id, None)
                    if hasattr(self, 'score_history'):
                        self.score_history.pop(old_id, None)
        
        for object_id, centroid in objects.items():
            if self.tracker.disappeared.get(object_id, 0) > 0:
                continue # Skip tracking actions for temporarily lost objects to conserve GPU
                
            rx1, ry1, rx2, ry2 = [int(v) for v in self.tracker.object_rects[object_id]]
            best_rect = (rx1, ry1, rx2, ry2)
            
            state = self.track_states.get(object_id, "TRACKING")
            
            # Rate-limit the evaluations to prevent the queue from instantly flooding with the same faces
            if not hasattr(self, 'last_verify_time'):
                self.last_verify_time = {}
                
            now_time = time.time()
            last_checked = self.last_verify_time.get(object_id, 0)
            
            should_analyze = False
            if state == "TRACKING" and (now_time - last_checked > 0.3):
                should_analyze = True
            elif state in ["INTRUDER", "ANALYZING"] and (now_time - last_checked > 1.0):
                # Never re-analyze TRUSTED (saving crowd GPU cycles), but do re-analyze INTRUDERS periodically periodically
                should_analyze = True
            
            if should_analyze and best_rect is not None:
                self.last_verify_time[object_id] = now_time
                # We have tracked it, now push to recognition queue (Thread 3)
                rx1, ry1, rx2, ry2 = best_rect
                pad_x = int((rx2 - rx1) * 0.2)
                pad_y = int((ry2 - ry1) * 0.2)
                px1 = max(0, rx1 - pad_x)
                py1 = max(0, ry1 - pad_y)
                px2 = min(w, rx2 + pad_x)
                py2 = min(h, ry2 + pad_y)
                
                face_crop = frame[py1:py2, px1:px2].copy()
                if face_crop.size == 0 or self.is_blurry(face_crop, 20.0):
                    continue
                    
                self.track_states[object_id] = "ANALYZING"
                # Push face crop to recognition thread; Drop oldest if backlog is severely stalled
                if self.recognition_queue.full():
                    try:
                        self.recognition_queue.get_nowait()
                    except Exception:
                        pass
                        
                try:
                    self.recognition_queue.put_nowait((object_id, face_crop))
                except Exception:
                    # Ignore if queue gets immediately filled by another parallel operation
                    pass

    def recognition_process_thread(self):
        """ Thread 3 -> Recognition
        Takes cropped faces from queue, uses ArcFace embedding to match against DB. """
        while self.running:
            try:
                object_id, face_img = self.recognition_queue.get(timeout=1.0)
                self.recognize_worker(object_id, face_img)
            except queue.Empty:
                continue
            except Exception as e:
                logger.info(f"Recognition Thread Error: {e}")

    def recognize_worker(self, object_id, face_img):
        try:
            if time.time() < getattr(self, "safe_mode_until", 0):
                self.track_states[object_id] = "TRACKING"
                return
                
            h, w = face_img.shape[:2]
            
            # REQUIREMENT: Minimum 15x15 Face Size for distant CCTV cameras
            if w < 15 or h < 15:
                logger.info(f"[{object_id}] REJECT: resolution too low ({w}x{h})")
                self.track_states[object_id] = "TRACKING"
                return
                
            # REQUIREMENT: Medium blur threshold (5.0) to ensure good-quality images without rejecting standard webcam streams
            if self.is_blurry(face_img, threshold=5.0):
                logger.info(f"[{object_id}] REJECT: excessive motion blur")
                self.track_states[object_id] = "TRACKING"
                return

            # --- CCTV IMAGE ENHANCEMENT PIPELINE ---
            try:
                # [Optimization] Disabled AI Super Resolution (LapSRN) because it artificially alters facial textures, destroying SFace metrics.
                pass
                    
                # 1. CLAHE: Normalize severe shadows and extreme lighting differences
                l, a, b = cv2.split(cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                l_cl = clahe.apply(l)
                face_enhanced = cv2.cvtColor(cv2.merge((l_cl, a, b)), cv2.COLOR_LAB2BGR)
                
                # 2. Sharpening: Extract structural edges lost due to compression/poor clarity
                kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
                face_img = cv2.filter2D(face_enhanced, -1, kernel)
            except Exception as e:
                logger.info(f"[{object_id}] Face enhancement skipped: {e}")
                pass
            # --- END ENHANCEMENT ---
            
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Extract high-precision geometric landmarks dynamically from the tight facial track
            # YuNet often fails to detect faces when the face occupies 100% of the cropped image.
            # Upscaling the crop dramatically improves landmark detection rate on CCTV.
            try:
                face_img_large = cv2.resize(face_img, (160, 160))
                boxes, lms = self.align_detector.detect(face_img_large, return_landmarks=True)
                face_landmarks = None
                if lms and boxes:
                    center_x, center_y = 80, 80
                    best_dist = float('inf')
                    for box, lm in zip(boxes, lms):
                        bx1, by1, bx2, by2 = box[:4]
                        bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
                        dist = (bcx - center_x)**2 + (bcy - center_y)**2
                        if dist < best_dist:
                            best_dist = dist
                            # Scale landmarks back to original face_img dimensions
                            scale_x, scale_y = w / 160.0, h / 160.0
                            lm_scaled = lm.copy()
                            lm_scaled[:, 0] *= scale_x
                            lm_scaled[:, 1] *= scale_y
                            face_landmarks = lm_scaled
            except Exception:
                face_landmarks = None
            
            if face_landmarks is None:
                logger.info(f"[{object_id}] REJECT: High-confidence alignment landmarks not found. Skipping embedding to prevent history corruption.")
                self.track_states[object_id] = "TRACKING"
                return
            
            # Removed rigid geometric ratio locks to properly support top-down, tilted, and side-profile matching. 
            # High-confidence landmarks natively map correctly in cv2.warpAffine.
            
            # GPU Accelerated Affine Embedding via ONNX ArcFace
            emb = self.recognizer.get_embedding(face_img_rgb, face_landmarks)
            
            if emb is None:
                self.track_states[object_id] = "TRACKING"
                return
                
            logger.info("Embedding generated")
            
            # Strict Vector Search - Real-Time SFace ONNX Cosine Similarity
            # We enforce 0.45 with a strict Gap of 0.08 to minimize false rejections and false positives.
            best_match_id, best_similarity, match_strength = self.faiss_db.identify(emb, threshold=0.45, gap=0.08)
            
            with self.lock:
                current_label = self.track_labels.get(object_id, "Unknown")
                
                # Identity Lock: Once an identity is confirmed, maintain it 
                if current_label != "Unknown" and current_label not in ["TRUSTED", "INTRUDER", "INTRUDER ALERT"]:
                    if best_match_id != self.known_names.get(current_label, current_label):
                        if best_match_id == "Unknown" or (match_strength == "WEAK" and best_similarity >= 0.30):
                            logger.info(f"[{object_id}] Identity Lock: Maintained {current_label} despite CCTV fluctuation.")
                            self.track_states[object_id] = "TRUSTED"
                            return
                        else:
                            logger.info(f"[{object_id}] Identity Lock Overridden: Strong contradiction from {best_match_id}.")
                            self.track_labels[object_id] = "Unknown"
                            self.track_states[object_id] = "TRACKING"
                            if object_id in self.identity_history:
                                self.identity_history[object_id].clear()
                    else:
                        self.track_states[object_id] = "TRUSTED"
                        return

                # ROLLING HISTORY FOR CCTV
                if object_id not in self.identity_history:
                    self.identity_history[object_id] = []
                    
                self.identity_history[object_id].append(best_match_id)
                
                # STABILITY CHECK: Max fluctuation <= 0.40 for ALL mode (relaxed for CCTV angle changes)
                if hasattr(self, 'score_history'):
                    if object_id not in self.score_history:
                        self.score_history[object_id] = []
                    self.score_history[object_id].append(best_similarity)
                    if len(self.score_history[object_id]) > 10:
                        self.score_history[object_id].pop(0)
                    if len(self.score_history[object_id]) > 1:
                        score_diff = max(self.score_history[object_id]) - min(self.score_history[object_id])
                        if score_diff > 0.40:
                            logger.info(f"[{object_id}] Stability Fail: Score fluctuated by {score_diff:.4f} > 0.40. Rejecting.")
                            self.track_states[object_id] = "TRACKING"
                            return
                
                # Keep last 15 frames for temporal consistency buffer
                if len(self.identity_history[object_id]) > 15:
                    self.identity_history[object_id].pop(0)
                    
                history = self.identity_history[object_id]
                
                # Minimum 5 tracking frames for stable identity confirmation
                if len(history) >= 5:
                    recent_5 = history[-5:]
                    
                    # Count occurrences of the most common prediction inside the 5-frame window
                    counts = {}
                    for hid in recent_5:
                        counts[hid] = counts.get(hid, 0) + 1
                        
                    best_id = None
                    best_count = 0
                    for hid, count in counts.items():
                        if hid != "Unknown" and count > best_count:
                            best_count = count
                            best_id = hid
                            
                    # If an ID is matched across at least 3 out of 5 consecutive frames, confirm it!
                    if best_id is not None and best_count >= 3:
                        confirmed_id = best_id
                        logger.info(f"[{object_id}] CONFIRMED IDENTITY: {confirmed_id} [{match_strength} MODE] (Voted {best_count}/5)")
                        
                        # ----- DIVERSITY CHECK / AUTO REJECT MODE -----
                        active_visible_matching = 0
                        for active_id, active_label in self.track_labels.items():
                            if active_id in self.tracker.objects and getattr(self.tracker, 'disappeared', {}).get(active_id, 0) == 0 and active_id != object_id:
                                if active_label == self.known_names.get(confirmed_id, confirmed_id):
                                    active_visible_matching += 1
                                    
                        if active_visible_matching > 0:
                            logger.info(f"[CRITICAL] SAFE MODE: Multiple bodies assigned '{confirmed_id}'. Model bias detected.")
                            self.safe_mode_until = time.time() + 30
                            self.track_states[object_id] = "TRACKING"
                            return
                            
                        # Make labeling permanent
                        self.track_labels[object_id] = self.known_names.get(confirmed_id, confirmed_id)
                        self.track_states[object_id] = "TRUSTED"
                        
                        now = time.time()
                        if confirmed_id not in self.alert_cooldowns or (now - self.alert_cooldowns[confirmed_id]) > 300:
                            self.alert_cooldowns[confirmed_id] = now
                            logger.info(f"[Action] Marking attendance for {confirmed_id}")
                            mark_attendance(confirmed_id, face_img)
                            
                            # Continuous learning
                            try:
                                conn = get_db_connection()
                                cursor = conn.cursor()
                                cursor.execute("SELECT face_embedding FROM Employees WHERE employee_id=?", (confirmed_id,))
                                row = cursor.fetchone()
                                if row and row['face_embedding']:
                                    embs = json.loads(row['face_embedding'])
                                    embs.append(emb.tolist())
                                    if len(embs) > 50: embs = embs[-50:]
                                    cursor.execute("UPDATE Employees SET face_embedding=? WHERE employee_id=?", (json.dumps(embs), confirmed_id))
                                    conn.commit()
                                conn.close()
                                threading.Thread(target=self.load_embeddings_from_db, daemon=True).start()
                            except Exception as e:
                                logger.info(f"Continuous learning failed: {e}")
                    elif counts.get("Unknown", 0) >= 4:
                        # Only mark intruder if we persistently see Unknowns across 4 out of 5 sequential frames
                        logger.info(f"[{object_id}] Persistently Unknown. Marking Intruder.")
                        self.track_states[object_id] = "INTRUDER"
                        self._mark_intruder(object_id, face_img)
                        self.identity_history[object_id].clear()
                    else:
                        logger.info(f"[DEBUG] Verification in progress for {object_id} (Need 3 matches, got {best_count})")
                        self.track_states[object_id] = "TRACKING"
                else:
                    logger.info(f"[DEBUG] Collecting frames for {object_id} ({len(history)}/5)")
                    self.track_states[object_id] = "TRACKING"
                    
        except Exception as e:
            logger.info(f"Recognition error ID {object_id}: {e}")
            self.track_states[object_id] = "TRACKING"
            
    def _mark_intruder(self, object_id, face_img):
        self.track_labels[object_id] = "INTRUDER ALERT"
        if object_id not in self.alert_cooldowns or (time.time() - self.alert_cooldowns[object_id]) > 60:
            self.alert_cooldowns[object_id] = time.time()
            # Play beep instantly in main thread for quick feedback
            try:
                import winsound
                winsound.Beep(2000, 1000)
            except:
                pass
            # Pass image correctly to alerts pipeline
            threading.Thread(target=trigger_intruder_alert, args=(face_img,), daemon=True).start()
        
        # 5-MINUTE COOLDOWN CHECK FOR INTRUDERS (global)
        now = time.time()
        last_intruder = self.cooldown_cache.get("global_intruder", 0)
        if now - last_intruder < self.COOLDOWN_SECONDS:
            logger.info("[DEBUG] Skipped logging intruder (in 5-min combined cooldown).")
            return
            
        # Delegate all image saving, DB logic, memory cache, and alerts to alerts.py
        trigger_intruder_alert(face_img)
