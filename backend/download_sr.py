import os
import urllib.request

def download_model():
    weights_dir = os.path.join(os.path.dirname(__file__), 'models', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    url = "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb"
    model_path = os.path.join(weights_dir, "LapSRN_x4.pb")
    
    if not os.path.exists(model_path):
        print("Downloading LapSRN_x4.pb (Fast Super Resolution Model) from GitHub...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    else:
        print("Model currently exists.")

if __name__ == "__main__":
    download_model()
