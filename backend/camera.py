import cv2
import threading
import time
import json
import os
from recognition import RecognitionEngine
from utils.logger import get_logger

logger = get_logger("camera")

camera = None
output_frame = None
latest_frame = None
lock = threading.Lock()
engine = RecognitionEngine()
camera_should_run = True

def get_camera_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    config_path = os.path.join(project_dir, 'camera_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading camera config: {e}")
    return {"camera_name": "Webcam", "rtsp_url": "0", "location": "Local"}

threads_started = False

def capture_thread():
    """ 
    Thread 1: Camera Capture ONLY. Never blocks. 
    Continuously pulls the latest frame and stores it for the processing thread.
    Auto-reconnects on failure to simulate industrial uptime requirements.
    """
    global camera, latest_frame, camera_should_run
    
    frame_counter = 0
    backoff_time = 2.0
    
    while camera_should_run:
        try:
            os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "100"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            
            cam_conf = get_camera_config()
            rtsp_url = cam_conf.get("rtsp_url", "0")
            if str(rtsp_url).isdigit():
                rtsp_url = int(rtsp_url)
                logger.info(f"Connecting to Local Camera: {rtsp_url}")
            else:
                logger.info(f"Connecting to CCTV stream at {rtsp_url}")
            
            if isinstance(rtsp_url, int):
                temp_camera = cv2.VideoCapture(rtsp_url)
            else:
                temp_camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                
            camera = None
            if temp_camera.isOpened():
                temp_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera = temp_camera
            else:
                temp_camera.release()
                
            # Auto-discovery fallback: if configured camera fails, scan for any working local camera
            if camera is None:
                logger.warning(f"Connection to configured camera ({rtsp_url}) failed. Auto-detecting available cameras...")
                for i in range(10): # Scan common local camera indices
                    if isinstance(rtsp_url, int) and i == rtsp_url:
                        continue
                    try_cam = cv2.VideoCapture(i)
                    if try_cam.isOpened():
                        try_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # Verify we can actually read from it
                        ret, _ = try_cam.read()
                        if ret:
                            camera = try_cam
                            logger.info(f"Successfully auto-connected to fallback camera index: {i}")
                            break
                        else:
                            try_cam.release()
                    else:
                        try_cam.release()
            
            if camera is None or not camera.isOpened():
                logger.warning(f"No available cameras found. Retrying in {backoff_time:.1f} seconds...")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 1.5, 60.0)
                continue
                
            logger.info("Camera connected successfully. Resetting backoff interval.")
            backoff_time = 2.0
            
            while camera_should_run:
                if camera is None or not camera.isOpened():
                    logger.warning("Camera disconnected - reconnecting...")
                    break
                    
                success, frame = camera.read()
                if not success:
                    logger.warning("Camera unreadable - skipping frame and reconnecting...")
                    time.sleep(2)
                    break
                
                frame_counter += 1
                if frame_counter % 300 == 0:
                    logger.debug("Frame stream active (heartbeat).")
                    
                # Maintain aspect ratio to prevent face distortion and embedding corruption
                h, w = frame.shape[:2]
                aspect_ratio = w / h
                new_w = 640
                new_h = int(new_w / aspect_ratio)
                frame = cv2.resize(frame, (new_w, new_h))
                
                with lock:
                    latest_frame = frame
                    
        except Exception as e:
            logger.error(f"Camera error in capture loop: {e}", exc_info=True)
            time.sleep(backoff_time)
            backoff_time = min(backoff_time * 1.5, 60.0)
            
        if camera is not None:
            camera.release()
            camera = None

def processing_thread():
    """
    Thread 2: Face Processing ONLY.
    Updates states safely without blocking the UI. Now mathematically immune to CPU looping!
    """
    global latest_frame, camera_should_run
    
    last_processed_id = id(None)
    frame_count = 0
    
    while camera_should_run:
        try:
            frame_to_process = None
            with lock:
                # INSTANT FIX: Never process the exact same frame twice! Removes 100% CPU lockup
                if latest_frame is not None and id(latest_frame) != last_processed_id:
                    frame_to_process = latest_frame.copy()
                    last_processed_id = id(latest_frame)
                    
            if frame_to_process is not None:
                frame_count += 1
                # Skip to process at ~10 FPS max, drastically freeing up CPU for Recognition logic
                if frame_count % 3 != 0:
                    continue
                    
                engine.process_frame(frame_to_process)
            else:
                # Rest thread if no new frame arrived
                time.sleep(0.01)
                
        except Exception as e:
            logger.exception(f"Uncaught pipeline error in processing_thread: {e}")
            time.sleep(1)

def start_camera():
    global camera_should_run, threads_started
    if threads_started:
        return
        
    camera_should_run = True
    threads_started = True
    
    engine.start()
    
    t1 = threading.Thread(target=capture_thread)
    t1.daemon = True
    t1.start()
    
    t2 = threading.Thread(target=processing_thread)
    t2.daemon = True
    t2.start()

def generate_frames():
    global latest_frame, lock, camera_should_run, threads_started
    if not threads_started:
        start_camera()
        time.sleep(2)
        
    while True:
        encoded_image = None
        display_frame = None
        
        with lock:
            if latest_frame is not None:
                display_frame = latest_frame.copy()
        
        if display_frame is not None:
            # Draw bound boxes BEFORE recognition output / independently from AI lag
            try:
                engine.draw_annotations(display_frame)
            except Exception as e:
                pass
                
            flag, encoded_image_tmp = cv2.imencode(".jpg", display_frame)
            if flag:
                encoded_image = encoded_image_tmp
        
        if encoded_image is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                   bytearray(encoded_image) + b'\r\n')
        
        # UI streaming stream speed (~30 FPS)
        time.sleep(0.03)

def stop_camera():
    global camera_should_run
    camera_should_run = False

def get_camera_status():
    if camera_should_run:
        return "Connected"
    return "Disconnected"
