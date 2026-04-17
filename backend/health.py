import psutil
import shutil
from camera import get_camera_status
from database import get_db_connection

def get_system_health():
    health_data = {}
    
    # 1. Camera connect status
    health_data['camera'] = get_camera_status()
    
    # 2. Database connection
    try:
        conn = get_db_connection()
        conn.cursor().execute("SELECT 1")
        conn.close()
        health_data['database'] = "Connected"
    except Exception as e:
        health_data['database'] = f"Error: {e}"
        
    # 3. Disk Storage
    try:
        total, used, free = shutil.disk_usage("/")
        percent_used = (used / total) * 100
        health_data['disk_space_percent'] = round(percent_used, 1)
        
        status = "Healthy"
        if percent_used > 90:
            status = "Warning: Disk almost full"
        health_data['disk_status'] = status
    except:
        health_data['disk_space_percent'] = 0
        health_data['disk_status'] = "Unknown"
        
    # 4. CPU/Memory (Optional extra value)
    health_data['cpu_percent'] = psutil.cpu_percent()
    health_data['memory_percent'] = psutil.virtual_memory().percent
    
    return health_data
