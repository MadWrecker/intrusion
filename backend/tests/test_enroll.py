import requests
import json
import os

url = "http://127.0.0.1:8000/add_employee"
data = {
    "employee_id": "TEST_001",
    "name": "Test Employee",
    "department": "Testing",
    "phone": "12345678"
}
files = [
    ("images", ("temp_face.jpg", open("../temp_face.jpg", "rb"), "image/jpeg"))
]

try:
    res = requests.post(url, data=data, files=files)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.text}")
except Exception as e:
    print(f"Error connecting to backend: {e}")
