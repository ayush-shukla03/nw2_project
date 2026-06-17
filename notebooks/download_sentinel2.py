import requests
import json

url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

filter_str = (
    "Collection/Name eq 'SENTINEL-2' "
    "and contains(Name,'MSIL1C') "
    "and ContentDate/Start gt 2023-07-01T00:00:00.000Z "
    "and ContentDate/Start lt 2023-08-31T00:00:00.000Z "
    "and OData.CSC.Intersects(area=geography'SRID=4326;"
    "POLYGON((89.70 25.80,95.50 25.80,95.50 27.80,89.70 27.80,89.70 25.80))')"
)

params = {
    "$filter": filter_str,
    "$orderby": "ContentDate/Start asc",
    "$top": 30,
}

print("Querying Copernicus catalogue for monsoon tiles...")
r = requests.get(url, params=params, timeout=60)
data = r.json()
products = data.get('value', [])

print(f"Found {len(products)} products\n")
for i, p in enumerate(products):
    name = p['Name']
    date = p['ContentDate']['Start'][:10]
    size = round(p['ContentLength']/1e9, 2)
    pid = p['Id']
    print(f"[{i}] {name}")
    print(f"     Date: {date} | Size: {size} GB | ID: {pid}")
    print()

with open('/mnt/nw2data/nw2_project/data/raw/sentinel2/monsoon_product_list.json', 'w') as f:
    json.dump(products, f, indent=2)
print("Product list saved.")