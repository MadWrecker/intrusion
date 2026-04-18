import os
import shutil
import time
import datetime
import json
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import jwt
from dotenv import load_dotenv

# =========================
# 🌍 ENV CHECK
# =========================
IS_RENDER = os.environ.get("RENDER") == "true"

# =========================
# 🟢 SAFE IMPORTS
# =========================
try:
    from apscheduler.schedulers.background import BackgroundScheduler
except:
    BackgroundScheduler = None

# =========================
# ⚠️ CONDITIONAL IMPORTS
# =========================
if not IS_RENDER:
    try:
        from health import get_system_health
    except:
        get_system_health = None

    try:
        from utils.logger import get_logger
    except:
        get_logger = None

    try:
        import cv2
        from models.recognizer import FastFaceRecognizer
        from models.detector import FastFaceDetector
    except:
        pass
else:
    # Safe fallbacks for Render
    def get_system_health():
        return {"status": "ok"}

    def get_logger(name):
        import logging
        return logging.getLogger(name)

# =========================
# DATABASE IMPORT
# =========================
from database import init_db, get_db_connection

# =========================
# INIT
# =========================
load_dotenv()
logger = get_logger("main") if get_logger else None

# =========================
# LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if logger:
        logger.info("Initializing system...")

    init_db()

    # ❗ Only run scheduler locally
    if not IS_RENDER and BackgroundScheduler:
        start_scheduler()

    yield

    if logger:
        logger.info("Shutting down system...")

    # ❗ Only stop camera locally
    if not IS_RENDER:
        try:
            import camera
            camera.stop_camera()
        except Exception:
            pass

# =========================
# APP INIT
# =========================
app = FastAPI(
    title="Factory AI Surveillance API",
    lifespan=lifespan
)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# AUTH CONFIG
# =========================
SECRET_KEY = os.environ.get("JWT_SECRET", "factory-guard-production-secret-key-2026")
ALGORITHM = "HS256"

def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        # Fallback to Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")




def mark_absences():
    """ Runs daily at 23:55 to mark absent employees """
    # Skip Sunday
    if datetime.datetime.now().weekday() == 6:
        return
        
    logger.info("Running Daily Absence Check...")
    today_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all employees
    cursor.execute("SELECT employee_id FROM Employees")
    employees = cursor.fetchall()
    
    # Get employees who have an attendance record today
    cursor.execute("SELECT employee_id FROM Attendance WHERE date=?", (today_date_str,))
    present_employees = {row['employee_id'] for row in cursor.fetchall()}
    
    for emp_row in employees:
        emp_id = emp_row['employee_id']
        if emp_id not in present_employees:
            cursor.execute('''
                INSERT INTO Attendance (employee_id, date, status)
                VALUES (?, ?, ?)
            ''', (emp_id, today_date_str, 'LEAVE'))
            
    conn.commit()
    conn.close()
    logger.info("Daily absence check completed.")

def start_scheduler():
    scheduler = BackgroundScheduler()
    # Run every day at 18:00 (6:00 PM) to mark ABSENT instead of LEAVE
    scheduler.add_job(mark_absences, 'cron', hour=18, minute=0)
    scheduler.start()

class LoginRequest(BaseModel):
    username: str
    password: str

class TempPassCreate(BaseModel):
    name: str
    purpose: str = ""
    image: str = ""

class TempPassStatusUpdate(BaseModel):
    status: str

@app.post("/login")
def login(req: LoginRequest, response: Response):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    hashed_password = hashlib.sha256(req.password.encode()).hexdigest()
    
    cursor.execute("SELECT id, username FROM Admin WHERE username=? AND password=?", (req.username, hashed_password))
    user = cursor.fetchone()
    
    # Handle migration from legacy plaintext password if they haven't re-initialized the DB
    if not user:
        cursor.execute("SELECT id, username FROM Admin WHERE username=? AND password=?", (req.username, req.password))
        user_legacy = cursor.fetchone()
        if user_legacy:
            user = user_legacy
            cursor.execute("UPDATE Admin SET password=? WHERE id=?", (hashed_password, user_legacy['id']))
            conn.commit()
            
    conn.close()
    
    if user:
        # Generate JWT Token
        expire_time = datetime.datetime.utcnow() + datetime.timedelta(hours=12)
        payload = {"sub": user["username"], "exp": expire_time}
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        
        # Set as HttpOnly cookie for seamless frontend integration
        is_prod = os.environ.get("ENVIRONMENT", "production").lower() == "production"
        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=is_prod,
            samesite="strict",
            max_age=43200 # 12 hours
        )
        logger.info(f"Admin logged in securely: {req.username}")
        return {"success": True, "token": token}
        
    logger.warning(f"Failed login attempt for username: {req.username}")
    raise HTTPException(status_code=401, detail="Invalid credentials")


if not IS_RENDER:

    @app.get("/camera_feed", dependencies=[Depends(get_current_user)])
    def camera_feed():
        from camera import generate_frames
        return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/camera/start", dependencies=[Depends(get_current_user)])
    def camera_start():
        import camera
        camera.camera_should_run = True
        return {"message": "Camera active"}

    @app.get("/camera/stop", dependencies=[Depends(get_current_user)])
    def camera_stop():
        import camera
        camera.camera_should_run = False
        return {"message": "Camera stopped"}

    @app.get("/camera/status", dependencies=[Depends(get_current_user)])
    def camera_status():
        from camera import get_camera_status
        return {"status": get_camera_status()}

@app.get("/alerts", dependencies=[Depends(get_current_user)])
def get_alerts():
    return get_active_alerts()

@app.get("/system_health", dependencies=[Depends(get_current_user)])
def system_health():
    return get_system_health()

@app.post("/add_employee", dependencies=[Depends(get_current_user)])
def add_employee(employee_id: str = Form(...), name: str = Form(...), department: str = Form(...), phone: str = Form(""), images: list[UploadFile] = File(...)):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    import numpy as np
    import cv2
    from camera import engine
    
    # Store ONLY embeddings in database (not raw images) as requested
    global global_detector, global_recognizer
    if 'global_detector' not in globals() or global_detector is None:
        global_detector = FastFaceDetector()
    if 'global_recognizer' not in globals() or global_recognizer is None:
        global_recognizer = FastFaceRecognizer()
        
    detector = global_detector
    recognizer = global_recognizer
    embeddings = []
    last_error = "Unknown failure"
    
    for idx, image_file in enumerate(images):
        if len(embeddings) >= 5:
            break # High-speed skip: we already have ample identity data for this person.
            
        try:
            # Read from memory directly
            file_bytes = np.frombuffer(image_file.file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to read the uploaded image.")
                
            # SPEED OPTIMIZATION & DETECTION FIX: Resize huge images
            # Face detectors struggle with 4K resolution and memory/CPU explodes.
            max_dim = 800
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            # ArcFace is incredibly robust to pose changes. Augmentations are no longer necessary and cause massive UI lag.
            augmented_images = [img]
            
            raw_embeddings = []
            
            for aug_idx, aug_img in enumerate(augmented_images):
                try:
                    # Detect face with our Fast ONNX model to perfectly match live camera
                    boxes, landmarks_list = detector.detect(aug_img, return_landmarks=True)
                    
                    if not boxes:
                        continue # Skip failed augmentations
                        
                    # If multiple faces detected, intelligently pick the LARGEST face (the subject in the photo)
                    if len(boxes) > 1:
                        best_idx = 0
                        max_area = 0
                        for i, box in enumerate(boxes):
                            area = (box[2] - box[0]) * (box[3] - box[1])
                            if area > max_area:
                                max_area = area
                                best_idx = i
                                
                        x1, y1, x2, y2, score = boxes[best_idx]
                        face_landmarks = landmarks_list[best_idx] if landmarks_list else None
                    else:
                        x1, y1, x2, y2, score = boxes[0]
                        face_landmarks = landmarks_list[0] if landmarks_list else None
                        
                    # Phase 1: Input Validation - be more permissive during explicit enrollment to prevent false failures
                    if score < 0.30:
                        continue
                    
                    # Crop exactly like live recognition (20% padding!)
                    h, w = aug_img.shape[:2]
                    
                    pad_x = int((x2 - x1) * 0.2)
                    pad_y = int((y2 - y1) * 0.2)
                    
                    px1 = max(0, int(x1) - pad_x)
                    py1 = max(0, int(y1) - pad_y)
                    px2 = min(w, int(x2) + pad_x)
                    py2 = min(h, int(y2) + pad_y)
                    
                    face_img = aug_img[py1:py2, px1:px2].copy()
                    if face_img.size == 0 or face_img.shape[0] < 25 or face_img.shape[1] < 25:
                        continue # Reject poor boxes
                        
                    # Instantly adjust landmarks from first detection instead of redundantly running YuNet a second time!
                    adjusted_landmarks = face_landmarks.copy() if face_landmarks is not None else None
                    if adjusted_landmarks is not None:
                        adjusted_landmarks[:, 0] -= px1
                        adjusted_landmarks[:, 1] -= py1

                    # Remove slow/heavy LapSRN for backend enrollments - only high res images should be used.
                    try:
                        l, a, b = cv2.split(cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB))
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                        l_cl = clahe.apply(l)
                        face_enhanced = cv2.cvtColor(cv2.merge((l_cl, a, b)), cv2.COLOR_LAB2BGR)
                        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
                        face_img = cv2.filter2D(face_enhanced, -1, kernel)
                    except Exception:
                        pass
                    # Recognizer expects RGB
                    face_img_rgb_final = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    emb = recognizer.get_embedding(face_img_rgb_final, adjusted_landmarks)
                    if emb is None:
                        continue
                    
                    # Ensure it is normalized float array
                    emb = np.array(emb, dtype=np.float32).flatten()
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                        raw_embeddings.append(emb.tolist())
                        
                    if len(raw_embeddings) >= 1:
                        break # Ultra-fast skip: we only need 1 embedding per image now
                        
                        
                except Exception as e:
                    last_error = f"Augmentation/Recognition Error: {e}"
                    print(f"[Augmentation Error] {e}")
                    pass
                    
            if len(raw_embeddings) < 1:
                if last_error == "Unknown failure":
                    last_error = f"Quality filter failed for image {idx}: Only generated {len(raw_embeddings)} valid embeddings."
                else:
                    last_error = f"Quality filter failed for image {idx} (Inner Exception: {last_error})"
                print(last_error)
                continue # Gracefully skip poor images instead of aborting EVERYTHING
                
            # EMBEDDING CLEANING (STEP 6) - Remove outliers far from cluster center
            raw_embeddings_np = np.array(raw_embeddings)
            centroid = np.mean(raw_embeddings_np, axis=0)
            centroid_norm = centroid / np.linalg.norm(centroid)
            
            # Keep consistent embeddings (similarity to centroid > 0.50)
            similarities = np.dot(raw_embeddings_np, centroid_norm)
            
            cleaned_embeddings = [raw_embeddings[i] for i, sim in enumerate(similarities) if sim > 0.50]
            
            if len(cleaned_embeddings) < 1:
                print(f"Image {idx} failed outlier removal.")
                continue # Skip gracefully
                
            embeddings.extend(cleaned_embeddings)
                
        except Exception as e:
            last_error = str(e)
            print(f"Embedding extraction skipped for image {idx}: {e}")
            pass # Skip image on error rather than aborting the whole signup process!
            
    if len(embeddings) < 1:
        raise HTTPException(status_code=400, detail=f"Failed to capture enough valid face data (got {len(embeddings)}). Internal Error context: {last_error}")
        
            
    embeddings_json = json.dumps(embeddings) if embeddings else None
        
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Remove existing employee with same ID before replacing
        cursor.execute("DELETE FROM Employees WHERE employee_id=?", (employee_id,))
        
        # Store ONLY embeddings in database (no raw images, empty image_folder)
        cursor.execute("INSERT INTO Employees (employee_id, name, department, phone, face_embedding, image_folder) VALUES (?, ?, ?, ?, ?, ?)", 
                       (employee_id, name, department, phone, embeddings_json, ""))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=str(e))
        
    conn.close()
    
    # CRITICAL: Hot-reload embeddings into the live tracking engine's active memory!
    try:
        from camera import engine
        if engine:
            engine.load_embeddings_from_db()
    except Exception as reload_err:
        logger.error(f"Error reloading embeddings into memory: {reload_err}")
        
    return {"success": True, "message": "Employee added successfully. Embeddings active immediately."}

@app.get("/employees", dependencies=[Depends(get_current_user)])
def get_employees():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, employee_id, name, department, phone, date_added FROM Employees")
    people = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return people

@app.delete("/employee/{employee_id}", dependencies=[Depends(get_current_user)])
def delete_employee(employee_id: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT image_folder FROM Employees WHERE employee_id=?", (employee_id,))
    row = cursor.fetchone()
    if row and row['image_folder'] and os.path.exists(row['image_folder']):
        try:
            shutil.rmtree(row['image_folder'])
        except:
            pass
            
    cursor.execute("DELETE FROM Employees WHERE employee_id=?", (employee_id,))
    conn.commit()
    conn.close()
            
    # Dynamically update the recognition engine if it's running
    try:
        from camera import engine
        if engine:
            engine.load_embeddings_from_db()
    except Exception as e:
        logger.error(f"Could not reload running engine: {e}")
            
    return {"success": True}

@app.get("/attendance", dependencies=[Depends(get_current_user)])
def get_attendance():
    """
    Returns today's attendance records by default, or could take a ?date= filter.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    today_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.id, e.name AS employee_name, a.employee_id, a.date, a.check_in_time, a.check_out_time, a.status, a.image_path 
        FROM Attendance a 
        JOIN Employees e ON a.employee_id = e.employee_id
        ORDER BY a.id DESC
    ''')
    records = []
    for row in cursor.fetchall():
        row_dict = dict(row)
        img_path = row_dict.get('image_path')
        if img_path:
            full_path = os.path.join(base_dir, img_path)
            if not os.path.exists(full_path):
                row_dict['image_path'] = None
        records.append(row_dict)
    conn.close()
    return records

@app.get("/intruder_logs", dependencies=[Depends(get_current_user)])
def get_intruder_logs():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM IntruderLogs ORDER BY id DESC LIMIT 100")
    logs = []
    for row in cursor.fetchall():
        row_dict = dict(row)
        # old legacy format uses db_relative_path like "intruder_detected/filename"
        img_path = row_dict.get('image_path')
        if img_path:
            full_path = os.path.join(base_dir, img_path)
            if os.path.exists(full_path):
                logs.append(row_dict)
    conn.close()
    return logs

@app.delete("/intruder_logs/{log_id}", dependencies=[Depends(get_current_user)])
def delete_intruder_log(log_id: int):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM IntruderLogs WHERE id=?", (log_id,))
    row = cursor.fetchone()
    if row and row['image_path']:
        try:
            # Check absolute path first, then relative to base
            img_path = row['image_path']
            if not os.path.isabs(img_path):
                img_path = os.path.join(base_dir, img_path)
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception as e:
            logger.error(f"Error removing intruder image: {e}")
            
    conn.commit()
    conn.close()
    return {"success": True}

# ==========================================
# 🛑 TEMPORARY PASS ROUTES
# ==========================================

@app.get("/temp-passes", dependencies=[Depends(get_current_user)])
def get_temp_passes():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM temporary_passes")
    passes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return passes

@app.post("/temp-passes", dependencies=[Depends(get_current_user)])
def create_temp_pass(temp_pass: TempPassCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    # If the user is unauthenticated, get_current_user will block.
    # We could make POST /temp-passes public if visitors request it themselves, but logic implies logged in.
    cursor.execute(
        "INSERT INTO temporary_passes (name, purpose, status, image) VALUES (?, ?, 'pending', ?)",
        (temp_pass.name, temp_pass.purpose, temp_pass.image)
    )
    conn.commit()
    conn.close()
    return {"message": "Temporary pass requested successfully. Waiting for approval."}

@app.put("/temp-passes/{pass_id}/status", dependencies=[Depends(get_current_user)])
def update_temp_pass_status(pass_id: int, status_update: TempPassStatusUpdate):
    if status_update.status not in ["pending", "approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE temporary_passes SET status = ? WHERE id = ?",
        (status_update.status, pass_id)
    )
    conn.commit()
    conn.close()
    return {"message": f"Temporary pass updated to {status_update.status}."}

@app.delete("/temp-passes/{pass_id}", dependencies=[Depends(get_current_user)])
def delete_temp_pass(pass_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM temporary_passes WHERE id = ?", (pass_id,))
    conn.commit()
    conn.close()
    return {"message": "Temporary pass deleted successfully"}

# Mount directories for static assets
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(base_dir, 'frontend')
vendor_dir = os.path.join(frontend_dir, 'vendor')
intruder_detected_dir = os.path.join(base_dir, 'intruder_detected')
employee_detected_dir = os.path.join(base_dir, 'employee_detected')

os.makedirs(frontend_dir, exist_ok=True)
os.makedirs(vendor_dir, exist_ok=True)
os.makedirs(intruder_detected_dir, exist_ok=True)
os.makedirs(employee_detected_dir, exist_ok=True)

app.mount("/intruder_detected", StaticFiles(directory=intruder_detected_dir), name="intruder_detected")
app.mount("/employee_detected", StaticFiles(directory=employee_detected_dir), name="employee_detected")
app.mount("/vendor", StaticFiles(directory=vendor_dir), name="vendor")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
