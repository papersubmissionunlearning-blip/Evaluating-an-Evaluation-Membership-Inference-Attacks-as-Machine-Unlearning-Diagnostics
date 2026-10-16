import os
import requests
from tqdm import tqdm
from zipfile import ZipFile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(BASE_DIR, "data", "MUCAC2")
os.makedirs(TARGET_DIR, exist_ok=True)


FILES = {
    "CelebAMask-HQ.zip": "https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/Eb37jNPPA7hHl0fmktYqcV8B-qmPLx-ZKYQ1eFk4UPBV_A?download=1",
    "CelebA-HQ-identity.txt": "https://postechackr-my.sharepoint.com/:t:/g/personal/dongbinna_postech_ac_kr/EVRoUY8_txRFv56-KWvZrksBDWbD6adkjBxwwRN7qAC6bg?download=1",
    "CelebA-HQ-attribute.txt": "https://postechackr-my.sharepoint.com/:t:/g/personal/dongbinna_postech_ac_kr/EVrdIrPOkR1OlEWBVK8lE3AB9bFh741GnKBkNgPa8trNuA?download=1"
}

def download_with_progress(url, filename):
    dest_path = os.path.join(TARGET_DIR, filename)
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=filename,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

for filename, url in FILES.items():
    print(f"Starting download: {filename}")
    download_with_progress(url, filename)

zip_path = os.path.join(TARGET_DIR, "CelebAMask-HQ.zip")
print("Extracting ZIP...")
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(TARGET_DIR)

print("Done! Files saved in:", TARGET_DIR)
