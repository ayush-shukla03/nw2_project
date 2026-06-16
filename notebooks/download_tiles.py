import requests
import os

# Your Copernicus credentials
USERNAME = "f20240719@goa.bits-pilani.ac.in"
PASSWORD = "Taru@8302700600"

OUTPUT_DIR = "/mnt/nw2data/nw2_project/data/raw/sentinel2/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get access token
print("Getting access token...")
token_resp = requests.post(
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    data={
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
    },
    timeout=30
)
token = token_resp.json()["access_token"]
print("Token obtained\n")

# Tile IDs to download (March 2nd, good coverage tiles)
tiles = [
    ("T46RDR", "0e3f8bae-e635-4a60-8a93-b42b3a6ba0dc"),
    ("T46RDP", "7776c956-f4a5-46c7-8ff8-5f99eb0f5d32"),
    ("T46RCP", "85d233f1-3f25-452e-a21e-e48263468c84"),
    ("T46RBQ", "871e974c-ebc5-4b0a-95f9-8d302e454734"),
    ("T46RDQ", "901eab77-04d7-4776-a6cf-100b7a770c2c"),
    ("T46RBR", "962c237e-baa2-435c-83a7-2bf89a88b1ac"),
    ("T46RBP", "9f14c1b6-8914-4e5c-a93d-d4ff4873f0f4"),
    ("T46RCR", "bc4c1ff2-6613-4a3c-9d1c-4a11044fb7bd"),
    ("T46RCQ", "f6128fe4-5647-40cf-9e24-b5db58723c5c"),
]

headers = {"Authorization": f"Bearer {token}"}

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {token}"})

for name, pid in tiles:
    out_path = os.path.join(OUTPUT_DIR, f"{name}.zip")
    if os.path.exists(out_path):
        print(f"[SKIP] {name} already exists")
        continue

    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"
    print(f"Downloading {name}...")

    with session.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                done = int(50 * downloaded / total) if total else 0
                print(f"\r  [{'█'*done}{'.'*(50-done)}] {downloaded/1e6:.1f}/{total/1e6:.1f} MB", end='')
        print(f"\n  ✓ Saved to {out_path}\n")

print("All downloads complete.")