import os
import urllib.request

def download_tailwind():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendor_dir = os.path.join(base_dir, 'frontend', 'vendor')
    os.makedirs(vendor_dir, exist_ok=True)
    
    tailwind_path = os.path.join(vendor_dir, 'tailwind-cdn.js')
    if not os.path.exists(tailwind_path):
        print("Downloading TailwindCSS for offline support...")
        url = "https://cdn.tailwindcss.com"
        try:
            urllib.request.urlretrieve(url, tailwind_path)
            print("Successfully downloaded TailwindCSS to", tailwind_path)
        except Exception as e:
            print("Failed to download TailwindCSS:", e)
    else:
        print("TailwindCSS already exists locally.")

if __name__ == "__main__":
    download_tailwind()
