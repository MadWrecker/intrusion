import os
import time
import requests
from datetime import datetime
from database import get_db_connection

IS_RENDER = os.environ.get("RENDER") == "true"

if not IS_RENDER:
    import cv2

try:
    import winsound
except ImportError:
    winsound = None

# In-memory storage for the latest alerts to feed the dashboard quickly
_active_alerts = []

def get_telegram_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'telegram_config.json')
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""), "chat_id": os.environ.get("TELEGRAM_CHAT_ID", "")}

def send_telegram_alert(image_path, camera_id, timestamp):
    config = get_telegram_config()
    token = config.get("bot_token", "")
    chat_id = config.get("chat_id", "")
    
    if not token or not chat_id:
        return # Skip if not configured
        
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    caption = f"⚠️ *Intruder Detected*\n\n*Time:* {timestamp}\n*Location:* Factory Entrance\n*Camera:* {camera_id}"
    
    try:
        with open(image_path, 'rb') as photo:
            requests.post(url, data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'Markdown'}, files={'photo': photo}, timeout=5)
    except Exception as e:
        print(f"Telegram alert failed: {e}")

def trigger_intruder_alert(frame, camera_id="MAIN-GATE-01"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp_readable = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    filename = f"intruder_{camera_id}_{timestamp}.jpg"
    # Ensures files save to the exact directory mounted by FastAPI '/intruder_detected'
    filepath = os.path.join(base_dir, 'intruder_detected', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save snapshot
    cv2.imwrite(filepath, frame)
    
    # Play alert beep
    if winsound:
        try:
            winsound.Beep(2000, 1000) # Higher pitch for intruder
        except:
            pass
            
    # Send Telegram message in background
    import threading
    threading.Thread(target=send_telegram_alert, args=(filepath, camera_id, timestamp_readable)).start()
    
    # Log to DB
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Web-safe relative path
    db_relative_path = f"intruder_detected/{filename}"
    
    # New accurate database format
    cursor.execute('''
        INSERT INTO intruders (timestamp, camera_name, image_path)
        VALUES (?, ?, ?)
    ''', (str(int(time.time())), camera_id, db_relative_path))
    
    # Old legacy format
    cursor.execute('''
        INSERT INTO IntruderLogs (image_path, date, time, camera_location, status)
        VALUES (?, ?, ?, ?, ?)
    ''', (db_relative_path, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%I:%M %p"), camera_id, "UNRESOLVED"))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    
    alert = {
        "id": inserted_id,
        "camera_id": camera_id,
        "time": timestamp_readable,
        "status": "UNRESOLVED",
        "image_path": filename
    }
    
    _active_alerts.append(alert)
    # Keep only the latest 10 active alerts in memory
    if len(_active_alerts) > 10:
        _active_alerts.pop(0)

def get_active_alerts():
    return _active_alerts

def clear_alert(alert_id: int):
    global _active_alerts
    _active_alerts = [a for a in _active_alerts if a['id'] != alert_id]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE IntruderLogs SET status='RESOLVED' WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()

def log_trusted_person(person_id, name, camera_id="MAIN-GATE-01"):
    # Handled completely by attendance_manager now, kept function for backward compatibility if needed, doing nothing
    pass
