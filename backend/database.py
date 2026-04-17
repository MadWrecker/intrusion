import sqlite3
import os
import hashlib

DB_PATH = 'factory_system.db'

def get_db_connection():
    # SQLite requires timeouts for multi-threaded environments to avoid "database is locked" errors
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    # Enable WAL mode for high concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Create new folder structures
    os.makedirs(os.path.join(base_dir, 'employees'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'employee_detected'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'intruder_detected'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Admin table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Insert default admin if not exists
    cursor.execute("SELECT * FROM Admin WHERE username = 'admin@factory.com'")
    if not cursor.fetchone():
        hashed_pw = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute("INSERT INTO Admin (username, password) VALUES ('admin@factory.com', ?)", (hashed_pw,))

    # Employees table (replaces TrustedPersons)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            phone TEXT,
            face_embedding TEXT,
            image_folder TEXT,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    try:
        cursor.execute("ALTER TABLE Employees ADD COLUMN phone TEXT")
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute("ALTER TABLE Employees ADD COLUMN face_embedding TEXT")
    except sqlite3.OperationalError:
        pass
    
    # Attendance table (Legacy fallback if needed)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            date TEXT NOT NULL,
            check_in_time TEXT,
            check_out_time TEXT,
            status TEXT,
            image_path TEXT,
            FOREIGN KEY (employee_id) REFERENCES Employees (employee_id)
        )
    ''')
    
    try:
        cursor.execute("ALTER TABLE Attendance ADD COLUMN image_path TEXT")
    except sqlite3.OperationalError:
        pass

    # IntruderLogs table (Frontend Dashboard mapping)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS IntruderLogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            camera_location TEXT,
            status TEXT DEFAULT 'UNRESOLVED'
        )
    ''')

    # New accurate attendance table as requested
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            image_path TEXT
        )
    ''')
    
    try:
        cursor.execute("ALTER TABLE daily_attendance ADD COLUMN image_path TEXT")
    except sqlite3.OperationalError:
        pass

    # New accurate intruders table as requested
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS intruders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            camera_name TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    ''')

    # Temporary passes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS temporary_passes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            purpose TEXT,
            status TEXT DEFAULT 'pending',
            image TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("Factory Database and directories initialized successfully.")

if __name__ == '__main__':
    init_db()

