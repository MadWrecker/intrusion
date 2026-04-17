import datetime
import time
import winsound
import threading
import os
import cv2
from database import get_db_connection

# Cooldown dictionary
# Format: { 'employee_id': timestamp }
cooldown_cache = {}
COOLDOWN_SECONDS = 300 # 5 minutes

def is_sunday():
    return datetime.datetime.now().weekday() == 6

def play_beep():
    try:
        # play a short beep sound
        # frequency 1000Hz, duration 300ms
        threading.Thread(target=winsound.Beep, args=(1000, 300), daemon=True).start()
    except Exception:
        pass

def mark_attendance(employee_id, frame=None):
    """
    Marks attendance based on the rules.
    Returns the generated message.
    """
    # if is_sunday():
    #     return "Not marked: Today is Sunday (Holiday)"
        
    now = time.time()
    current_time_str = datetime.datetime.now().strftime("%I:%M %p")
    today_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    file_time_str = datetime.datetime.now().strftime("%H-%M-%S")
    
    # Check cooldown (duplicate prevention)
    if employee_id in cooldown_cache:
        if now - cooldown_cache[employee_id] < COOLDOWN_SECONDS:
            return f"Ignored: In cooldown for {employee_id}"
            
    # Fetch employee name
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM Employees WHERE employee_id=?", (employee_id,))
    emp_row = cursor.fetchone()
    
    if not emp_row:
        conn.close()
        return "Error: Employee not found."
    
    emp_name = emp_row['name']
    s_name = emp_name.replace(' ', '_')
    
    # Update cooldown early to avoid double-processing
    cooldown_cache[employee_id] = now
    
    try:
        # Check if attendance already exists for today
        cursor.execute("SELECT * FROM daily_attendance WHERE employee_id=? AND date=?", (employee_id, today_date_str))
        attendance_rows = cursor.fetchall()
        
        # Image formatting requested: employee_<name>_<date>_<time>.jpg
        image_filename = f"employee_{s_name}_{today_date_str}_{file_time_str}.jpg"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_save_path = os.path.join(base_dir, 'employee_detected', image_filename)
        db_relative_path = f"employee_detected/{image_filename}"
        
        if frame is not None:
            cv2.imwrite(full_save_path, frame)
        
        if len(attendance_rows) == 0:
            # First detection of the day -> ENTRY
            status = "ENTRY"
                
            cursor.execute('''
                INSERT INTO daily_attendance (employee_id, name, date, time, status, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (employee_id, emp_name, today_date_str, current_time_str, status, db_relative_path))
            
            # Also log to generic Attendance for backward compatibility with frontend dashboard
            cursor.execute('''
                INSERT INTO Attendance (employee_id, date, check_in_time, status, image_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (employee_id, today_date_str, current_time_str, "PRESENT", db_relative_path))
            
            conn.commit()
            return f"Welcome {emp_name} - {status} Recorded"
            
        elif len(attendance_rows) == 1:
            # Second detection of the day -> EXIT
            status = "EXIT"
            cursor.execute('''
                INSERT INTO daily_attendance (employee_id, name, date, time, status, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (employee_id, emp_name, today_date_str, current_time_str, status, db_relative_path))
            
            # Update legacy table checkout time
            cursor.execute('''
                UPDATE Attendance 
                SET check_out_time=? 
                WHERE employee_id=? AND date=? AND check_out_time IS NULL
            ''', (current_time_str, employee_id, today_date_str))
            
            conn.commit()
            return f"Goodbye {emp_name} - {status} Recorded"
        else:
            # Third or more -> Ignore
            return f"Ignored: Already completed Entry and Exit today for {emp_name}"
            
    except Exception as e:
        # Prevent silent failures from locking the user out for 5 minutes
        if employee_id in cooldown_cache:
            del cooldown_cache[employee_id]
        print(f"[ERROR] Attendance DB Crash: {e}")
        return f"Database error for {emp_name}: {e}"
        
    finally:
        conn.close()
