import requests
import os

# Your Copernicus credentials
USERNAME = "redacted" #put your username here
PASSWORD = "hehe not gonna tell you" #put your password here

OUTPUT_DIR = "/mnt/nw2data/nw2_project/data/raw/sentinel2/monsoon/"
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

# Tile IDs to download (July 2nd, good coverage tiles)
tiles = [
    ("monsoon_T46RFQ", "5ec595f5-f7cb-4dd4-9412-5aba47740711"),
    ("monsoon_T46RFP", "fe68bed8-5156-492e-8fc9-3bb66871ab44"),
    ("monsoon_T46REQ", "ffa6a53f-c835-478b-a779-6aef3b0158ca"),
    ("monsoon_T46RER", "b174c9f7-d846-45ce-8193-ee2e93c65989"),
    ("monsoon_T46RFR", "d6d2244f-7141-4d66-94f3-fb7a6e128e01"),
    ("monsoon_T46REP", "edacc56a-7d7a-44a9-a087-43dfa82557fe"),
    ("monsoon_T46RGR", "b38c186c-c93a-44ef-ae83-e25cb737e2ed"),
    ("monsoon_T46RDP", "52ad5718-a147-4abf-9a70-42c9501da8d6"),
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