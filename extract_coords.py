import urllib.request
import zipfile
import os
from pathlib import Path
import geopandas as gpd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
ZIP_PATH = DATA_DIR / "taxi_zones.zip"
EXTRACT_DIR = DATA_DIR / "taxi_zones"
OUT_CSV = DATA_DIR / "zone_coords.csv"

# Download the zip file
url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
if not ZIP_PATH.exists():
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, ZIP_PATH)

# Extract the zip
if not EXTRACT_DIR.exists():
    print("Extracting shapefile...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# Read shapefile and calculate centroids
print("Processing shapefile to get centroids...")
# The shapefile has a specific coordinate reference system (CRS) - typically EPSG:2263 for NYC/Long Island.
# We convert it to EPSG:4326 (standard lat/lon) to get standard geographic coordinates.
gdf = gpd.read_file(EXTRACT_DIR / "taxi_zones" / "taxi_zones.shp")
gdf = gdf.to_crs("EPSG:4326")

# Centroid calculations
# To avoid UserWarning about centroid calculation on geographic CRS, we can use EPSG:3857 for centroid, 
# then convert back. But for our rough ETA purposes, calculating on EPSG:4326 or EPSG:2263 directly is fine.
# Let's do it properly: calculate on original projected CRS (EPSG:2263), then convert to 4326.
gdf_orig = gpd.read_file(EXTRACT_DIR / "taxi_zones" / "taxi_zones.shp")
centroids = gdf_orig.geometry.centroid
centroids = centroids.to_crs("EPSG:4326")

gdf["longitude"] = centroids.x
gdf["latitude"] = centroids.y

# Keep only necessary columns: LocationID, latitude, longitude
df = gdf[["LocationID", "latitude", "longitude"]].copy()
df.rename(columns={"LocationID": "zone_id"}, inplace=True)
df.sort_values("zone_id", inplace=True)

# Save to CSV
df.to_csv(OUT_CSV, index=False)
print(f"Successfully saved zone coordinates to {OUT_CSV}")
