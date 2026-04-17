import urllib.request
import os

MODELS = {
    "backend/models/weights/yolov8n-face.onnx": "https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn/raw/main/weights/yolov8n-face.onnx",
    "backend/models/weights/arcface.onnx": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
    "backend/tracking/sort.py": "https://raw.githubusercontent.com/abewley/sort/master/sort.py"
}

def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"[*] Already exists: {target_path}")
        return
        
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"Downloading {url} to {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"[+] Successfully downloaded {target_path}")
    except Exception as e:
        print(f"[-] Failed to download {target_path}. Error: {e}")

if __name__ == "__main__":
    print("=== Downloading Required Models and Scripts ===")
    for path, url in MODELS.items():
        download_file(url, path)
    print("=== Download Complete ===")
