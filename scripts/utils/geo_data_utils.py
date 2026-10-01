# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Instituto de Investigaciones Agropecuarias (INIA), Chile
import io
import json
import logging
import math
from numbers import Rational
import os
import subprocess
import csv
import requests
import utm
import shutil


# GeoJSON generation function

from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
import piexif
from pytz import utc
from fractions import Fraction
from pyproj import Proj
from pillow_heif import register_heif_opener, open_heif
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import pyproj

# Lazy import rasterio to avoid dependency issues in GPS-only workflows
# import rasterio
# from rasterio.warp import transform
rasterio = None  # Will be imported when needed

register_heif_opener()

def clean_metadata(value):
    """Convert IFDRational, Fraction, or bytes to standard JSON-compatible types."""
    try:
        if isinstance(value, dict):
            return {k: clean_metadata(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [clean_metadata(item) for item in value]
        elif isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        elif isinstance(value, (Fraction, int, float)):
            return float(value)  # ✅ Convert IFDRational or Fraction to float
        elif isinstance(value, Rational):  # ✅ Ensure IFDRational is converted
            return float(value.numerator) / float(value.denominator)
        return value
    except Exception as e:
        print(f"❌ Error cleaning metadata: {str(e)}")
        return None

def convert_to_decimal_degrees(coord, ref=None, force_south=True):
    """Convert GPS coordinates from degrees, minutes, seconds to decimal degrees."""
    try:
        if not coord or not isinstance(coord, (list, tuple)) or len(coord) != 3:
            return None

        # Convert DMS (degrees, minutes, seconds) to decimal degrees
        coord = [float(x) if isinstance(x, (int, float)) else float(x.numerator) / float(x.denominator) 
                 for x in coord]
        
        decimal_value = coord[0] + coord[1] / 60 + coord[2] / 3600

        # ✅ Always force latitude to be negative (force_south=True)
        if force_south:
            decimal_value = -abs(decimal_value)  # Force negative latitude

        return decimal_value

    except Exception as e:
        print(f"❌ Error converting coordinates: {str(e)}")
        return None


def extract_heic_metadata_with_pillow_heif(image_path):
    """Extract metadata from HEIC files using pillow-heif."""
    try:
        heif_img = open_heif(image_path)
        if not hasattr(heif_img, 'metadata') or not heif_img.metadata:
            return None

        metadata = heif_img.metadata
        
        # Check if metadata contains 'exif' key
        if 'exif' not in metadata:
            # Try to parse raw metadata if available
            if hasattr(heif_img, 'raw_exif') and heif_img.raw_exif:
                # Create a temporary in-memory file with the EXIF data
                exif_bytes = heif_img.raw_exif
                temp_jpg = io.BytesIO(b'\xff\xd8\xff\xe1' + len(exif_bytes).to_bytes(2, byteorder='big') + exif_bytes)
                
                # Open with PIL to extract EXIF
                with Image.open(temp_jpg) as img:
                    exif_data = img._getexif()
                    if not exif_data:
                        return None
                        
                    # Extract GPS info
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        if tag_name == "GPSInfo":
                            gps_info = {}
                            for key in value.keys():
                                gps_info[GPSTAGS.get(key, key)] = value[key]
                            return gps_info
            return None
            
        # Extract GPS data from metadata if 'exif' key exists
        gps_info = {k: v for k, v in metadata.get('exif', {}).items() if 'GPS' in k}
        return gps_info if gps_info else None
        
    except Exception as e:
        print(f"❌ Error extracting HEIC metadata with pillow-heif: {str(e)}")
        return None


def extract_gps_with_exiftool(image_path):
    """Extract GPS coordinates from images using ExifTool."""
    try:
        if shutil.which("exiftool") is None:
            # ExifTool not available in this environment
            return None, None, None, None, None

        command = [
            "exiftool", "-j", "-n",
            "-GPSLatitude", "-GPSLongitude", "-GPSAltitude",
            "-GPSLatitudeRef", "-GPSLongitudeRef",
            image_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️ ExifTool error: {result.stderr}")
            return None, None, None, None, None
        
        exif_data = json.loads(result.stdout)
        if not exif_data or not exif_data[0]:
            return None, None, None, None, None

        data = exif_data[0]

        # ✅ Extract fields
        latitude = data.get("GPSLatitude")
        longitude = data.get("GPSLongitude")
        altitude = data.get("GPSAltitude", 0)
        lat_ref = data.get("GPSLatitudeRef", "S")  # Default South
        lon_ref = data.get("GPSLongitudeRef", "W")  # Default West

        print(f"✅ Extracted GPS → Lat: {latitude}, Lon: {longitude}, Alt: {altitude}, Ref: {lat_ref}/{lon_ref}")
        return latitude, longitude, altitude, lat_ref, lon_ref

    except Exception as e:
        print(f"❌ ExifTool failed: {str(e)}")
        return None, None, None, None, None


def extract_gps_with_exifread(image_path):
    """Extract GPS metadata from JPG/JPEG using exifread (robust when Pillow can't decode GPSInfo)."""
    try:
        import exifread  # optional dependency (present in yolov8_custom conda env)

        def _rat_to_float(r):
            return float(r.num) / float(r.den) if getattr(r, "den", 0) else 0.0

        def _dms_to_deg(values):
            d = _rat_to_float(values[0])
            m = _rat_to_float(values[1])
            s = _rat_to_float(values[2])
            return d + m / 60.0 + s / 3600.0

        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        if "GPS GPSLatitude" not in tags or "GPS GPSLongitude" not in tags:
            return None, None, None, None, None

        lat = _dms_to_deg(tags["GPS GPSLatitude"].values)
        lon = _dms_to_deg(tags["GPS GPSLongitude"].values)

        lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
        lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))
        if lat_ref != "N":
            lat = -abs(lat)
        if lon_ref != "E":
            lon = -abs(lon)

        altitude = None
        try:
            if "GPS GPSAltitude" in tags and tags["GPS GPSAltitude"].values:
                alt = tags["GPS GPSAltitude"].values[0]
                altitude = float(alt.num) / float(alt.den) if alt.den else 0.0
        except Exception:
            altitude = None

        return lat, lon, altitude if altitude is not None else 0, lat_ref, lon_ref

    except Exception:
        return None, None, None, None, None


def extract_gps_with_pil(image_path):
    """Extract GPS metadata from JPG/PNG using PIL."""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None, None, None, None, None

            gps_info = {}
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    for key in value.keys():
                        gps_info[ExifTags.GPSTAGS.get(key, key)] = value[key]

        return process_gps_info(gps_info)

    except Exception as e:
        print(f"❌ PIL extraction failed: {str(e)}")
        return None, None, None, None, None


def process_gps_info(gps_info):
    """Process extracted GPS info and convert to decimal degrees."""
    if not gps_info:
        return None, None, None, None, None

    lat_ref = gps_info.get("GPSLatitudeRef", "S")
    lon_ref = gps_info.get("GPSLongitudeRef", "W")

    latitude = convert_to_decimal_degrees(gps_info.get("GPSLatitude"), lat_ref)
    longitude = convert_to_decimal_degrees(gps_info.get("GPSLongitude"), lon_ref)
    altitude = gps_info.get("GPSAltitude", 0)

    return latitude, longitude, altitude, lat_ref, lon_ref


def convert_to_decimal_degrees(coord, ref):
    """Convert GPS coordinates from DMS to decimal degrees."""
    try:
        if not coord or not isinstance(coord, (list, tuple)) or len(coord) != 3:
            return None

        decimal_value = coord[0] + coord[1] / 60 + coord[2] / 3600

        # Ensure correct hemisphere
        if ref in ["S", "W"]:
            decimal_value = -abs(decimal_value)

        return decimal_value

    except Exception as e:
        print(f"❌ Error converting coordinates: {str(e)}")
        return None

def extract_gps_coordinates(image_path):
    """Extract GPS coordinates with robust fallbacks across environments."""
    try:
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # HEIC/HEIF: try ExifTool (best), then pillow-heif metadata
        if file_ext in (".heic", ".heif"):
            lat, lon, alt, lat_ref, lon_ref = extract_gps_with_exiftool(image_path)
            if lat is not None and lon is not None:
                return lat, lon, alt, lat_ref, lon_ref

            gps_info = extract_heic_metadata_with_pillow_heif(image_path)
            if gps_info:
                return process_gps_info(gps_info)

            return None, None, None, None, None

        # JPG/JPEG: Pillow sometimes fails to decode GPSInfo on some phones.
        if file_ext in (".jpg", ".jpeg"):
            lat, lon, alt, lat_ref, lon_ref = extract_gps_with_pil(image_path)
            if lat is not None and lon is not None:
                return lat, lon, alt, lat_ref, lon_ref

            lat, lon, alt, lat_ref, lon_ref = extract_gps_with_exifread(image_path)
            if lat is not None and lon is not None:
                return lat, lon, alt, lat_ref, lon_ref

            # Last resort if available
            return extract_gps_with_exiftool(image_path)

        # ✅ Use PIL for JPG/PNG if ExifTool fails
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None, None, None, None, None

            gps_info = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    for key in value.keys():
                        gps_info[GPSTAGS.get(key, key)] = value[key]

        lat_ref = gps_info.get("GPSLatitudeRef", "S")
        lon_ref = gps_info.get("GPSLongitudeRef", "W")
        latitude = convert_to_decimal_degrees(gps_info.get("GPSLatitude"), lat_ref)
        longitude = convert_to_decimal_degrees(gps_info.get("GPSLongitude"), lon_ref)

        # ✅ FIX ALTITUDE PARSING
        altitude_raw = gps_info.get("GPSAltitude", 0)
        try:
            if isinstance(altitude_raw, tuple):
                altitude = float(altitude_raw[0]) / float(altitude_raw[1]) if altitude_raw[1] != 0 else 0
            else:
                altitude = float(altitude_raw)
        except Exception:
            altitude = 0

        return latitude, longitude, altitude, lat_ref, lon_ref

    except Exception as e:
        print(f"❌ Error extracting GPS: {str(e)}")
        return None, None, None, None, None

def latlon_to_utm(lat, lon, lat_ref="S", lon_ref="W"):
    """
    Convert latitude and longitude to UTM coordinates.
    The UTM zone number is determined based on longitude.
    The hemisphere (N/S) is inferred from latitude.
    """
    if lat is None or lon is None:
        return None, None, None

    utm_easting, utm_northing, utm_zone, hemisphere = utm.from_latlon(lat, lon)
    utm_zone_label = f"{utm_zone}{'S' if lat < 0 else 'N'}"

    return utm_easting, utm_northing, utm_zone_label

def generate_geojson(image_name, metadata, output_dir):
    """
    Generates a JSON file with the model information, bounding boxes, video metadata, and class summary.
    Saves the JSON file to the specified json_output_path for videos.

    Args:
    video_metadata (dict): Metadata of the video.
    bboxes (list): List of bounding boxes and labels.
    model_info (dict): Information about the model and its configuration.
    label_summary (Counter): A summary of detected objects by class.
    json_output_path (str): Full file path to save the JSON output.

    Returns:
    None
    """
    if not metadata:
        print(f"⚠️ Missing metadata for {image_name}, skipping GeoJSON creation.")
        return

    # Extract fields safely
    gps = metadata.get("GPS_Coordinates", {})
    utm = metadata.get("UTM_Coordinates", {})
    model = metadata.get("model_info", {})

    latitude = gps.get("Latitude", "")
    longitude = gps.get("Longitude", "")
    utm_easting = utm.get("UTM_Easting", "")
    utm_northing = utm.get("UTM_Northing", "")
    zone = utm.get("Zone", "")
    altitude = utm.get("Altitude", "")
    terrain = utm.get("Terrain_Elevation", "")
    agl = utm.get("AGL", "")
    fov_area = utm.get("FOV_Area_m2", "")

    # Enrich metadata explicitly
    metadata["GPS_Coordinates"]["Latitude"] = latitude
    metadata["GPS_Coordinates"]["Longitude"] = longitude
    metadata["UTM_Coordinates"]["UTM_Easting"] = utm_easting
    metadata["UTM_Coordinates"]["UTM_Northing"] = utm_northing
    metadata["UTM_Coordinates"]["Zone"] = zone
    metadata["UTM_Coordinates"]["Altitude"] = altitude
    metadata["UTM_Coordinates"]["Terrain_Elevation"] = terrain
    metadata["UTM_Coordinates"]["AGL"] = agl
    metadata["UTM_Coordinates"]["FOV_Area_m2"] = fov_area

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": metadata,
                "geometry": {
                    "type": "Point",
                    "coordinates": [longitude, latitude] if latitude != "" and longitude != "" else []
                }
            }
        ]
    }

    json_output_path = os.path.join(output_dir, "JSON_metadata", f"{image_name}_image_prediction_metadata.json")
    with open(json_output_path, 'w') as f:
        json.dump(geojson_data, f, indent=4)
    print(f"✅ GeoJSON metadata saved for {image_name}: {json_output_path}")


def gather_unique_labels(batch_dir):
    """
    Gather all unique labels from the batch of images by scanning the Label_Summary fields
    in the metadata files.

    Args:
    batch_dir (str): The directory containing the batch of images with JSON metadata.

    Returns:
    set: A set of unique labels/classes found in all images.
    """
    unique_labels = set()

    # Traverse through the batch directory to find JSON metadata files
    for root, dirs, files in os.walk(batch_dir):
        for file in files:
            if file.endswith('_image_metadata.json'):
                json_metadata_path = os.path.join(root, file)
                with open(json_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    label_summary = metadata['label_summary']
                    unique_labels.update(label_summary.keys())  # Add all keys (labels) to the set

    return unique_labels


def generate_batch_summary_csv(batch_dir, output_csv, results):
    """Generate CSV with UTM coordinates, labels, image paths, and processing times."""
    unique_labels = set()
    for result in results:
        unique_labels.update(result['label_summary'].keys())
    unique_labels = sorted(unique_labels)

    # CSV Header
    csv_header = ['Image_Name', 'UTM_Easting', 'UTM_Northing', 'Zone', 'Altitude', 'Image_Path', 'Styled_Image_Path', 'Processing_Time'] + unique_labels

    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)

        for result in results:
            row_data = [
                result['image_name'],
                result['utm_easting'],
                result['utm_northing'],
                result['zone'],
                result['altitude'],
                result['image_path'],
                result['styled_image_path'],
                result['processing_time']
            ]

            # Add label counts (default to 0 if not present)
            for label in unique_labels:
                row_data.append(result['label_summary'].get(label, 0))

            csv_writer.writerow(row_data)

    print(f"📂 CSV summary generated: {output_csv}")


def extract_metadata(image_path):
    """Extract EXIF metadata from an image and clean it before returning."""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return {}

            metadata = {}
            gps_data = {}

            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                metadata[tag_name] = clean_metadata(value)  # ✅ Apply cleaning

                # Extract and clean GPS metadata separately
                if tag_name == "GPSInfo":
                    for gps_tag in value:
                        sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                        gps_data[sub_tag] = clean_metadata(value[gps_tag])  # ✅ Also clean GPS data

            metadata["GPSInfo"] = gps_data
            return clean_metadata(metadata)  # ✅ Ensure entire metadata is cleaned

    except Exception as e:
        print(f"❌ Error extracting metadata from {image_path}: {str(e)}")
        return {}


def generate_qgis_summary_csv(output_dir):
    """Generate CSV summary using original GPS metadata stored in JSON files, including UTM coordinates and LYPES-T."""

    json_metadata_dir = os.path.join(output_dir, "JSON_metadata")
    if not os.path.exists(json_metadata_dir):
        print(f"❌ Error: `JSON_metadata` folder not found in {output_dir}.")
        return

    results = []
    unique_labels = set()
    contains_lypes = False

    for filename in os.listdir(json_metadata_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(json_metadata_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    properties = data.get("features", [{}])[0].get("properties", {})

                    image_name = properties.get("image_name", "Unknown")
                    image_path = properties.get("image_path", "Unknown")
                    styled_image_path = properties.get("styled_image_link", "Unknown")

                    gps_data = properties.get("GPS_Coordinates", {})
                    latitude = gps_data.get("Latitude", None)
                    longitude = gps_data.get("Longitude", None)

                    utm_data = properties.get("UTM_Coordinates", {})
                    utm_easting = utm_data.get("UTM_Easting", None)
                    utm_northing = utm_data.get("UTM_Northing", None)
                    zone = utm_data.get("Zone", None)
                    altitude = utm_data.get("Altitude", None)
                    terrain = utm_data.get("Terrain_Elevation", None)
                    offset = utm_data.get("Offset", None)
                    agl = utm_data.get("AGL", None)
                    fov_area = utm_data.get("FOV_Area_m2", None)

                    if latitude is None or longitude is None:
                        print(f"⚠️ Skipping {image_name} due to missing GPS coordinates.")
                        continue

                    model_info = properties.get("model_info", {})
                    model_version = model_info.get("model_version", "Unknown")
                    confidence_threshold = model_info.get("confidence_threshold", "Unknown")
                    img_size = model_info.get("img_size", "Unknown")
                    processing_time = properties.get("processing_time", "Unknown")

                    label_summary = properties.get("label_summary", {})
                    unique_labels.update(label_summary.keys())

                    lypes_labels = ["LYPES-G", "LYPES-O", "LYPES-R", "LYPES-Y"]
                    contains_lypes = any(label in label_summary for label in lypes_labels)
                    lypes_t_sum = sum(label_summary.get(label, 0) for label in lypes_labels) if contains_lypes else None

                    results.append({
                        "image_name": image_name,
                        "latitude": latitude,
                        "longitude": longitude,
                        "utm_easting": utm_easting,
                        "utm_northing": utm_northing,
                        "zone": zone,
                        "altitude": altitude,
                        "terrain": terrain,
                        "agl": agl,
                        "offset": offset,
                        "fov_area": fov_area,
                        "image_path": image_path,
                        "styled_image_path": styled_image_path,
                        "processing_time": processing_time,
                        "model_version": model_version,
                        "confidence_threshold": confidence_threshold,
                        "img_size": img_size,
                        "label_summary": label_summary,
                        "LYPES-T": lypes_t_sum
                    })

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")

    unique_labels = sorted(unique_labels)
    output_csv_path = os.path.join(output_dir, "QGIS_summary.csv")

    csv_header = [
        "Image_Name", "Latitude", "Longitude",
        "UTM_Easting", "UTM_Northing", "Zone", "Altitude",
        "Terrain_Elevation", "AGL", "FOV_Area_m2",
        "Image_Path", "Styled_Image_Path", "Processing_Time", "Model_Version",
        "Confidence_Threshold", "Img_Size"
    ]

    if contains_lypes:
        csv_header.append("LYPES-T")

    csv_header += unique_labels

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)

        for result in results:
            row_data = [
                result["image_name"], result["latitude"], result["longitude"],
                result["utm_easting"], result["utm_northing"], result["zone"], result["altitude"],
                result["terrain"], result["agl"], result["fov_area"],
                result["image_path"], result["styled_image_path"], result["processing_time"],
                result["model_version"], result["confidence_threshold"], result["img_size"]
            ]

            if contains_lypes:
                row_data.append(result["LYPES-T"])

            for label in unique_labels:
                row_data.append(result["label_summary"].get(label, 0))

            csv_writer.writerow(row_data)

    print(f"✅ QGIS CSV summary saved: {output_csv_path}")


def generate_shapefile(csv_file, output_dir):
    """
    Convert YOLO inference results stored in CSV to a Shapefile (.shp) for QGIS integration.
    
    Args:
        csv_file (str): Path to the CSV file containing YOLO inference results.
        output_dir (str): Directory to save the output Shapefile.
    """
    try:
        df = pd.read_csv(csv_file)

        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            print(f"⚠️ Skipping {csv_file}: Missing required Latitude/Longitude columns.")
            return

        # Convert CSV to GeoDataFrame with points
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df['Longitude'], df['Latitude'])],
            crs="EPSG:4326"  # Define WGS84 (Lat/Lon)
        )

        # Convert to UTM (assumes Zone 19S, change as needed)
        gdf = gdf.to_crs(epsg=32719)  # EPSG:32719 → UTM Zone 19 South

        # Save as Shapefile
        shp_filename = os.path.join(output_dir, "inference_results.shp")
        gdf.to_file(shp_filename, driver='ESRI Shapefile')

        print(f"✅ Shapefile saved at {shp_filename}")

    except Exception as e:
        print(f"❌ Error creating shapefile: {str(e)}")




def copy_exif_metadata(original_image_path, predicted_image_path):
    """
    Copies both EXIF and XMP metadata from the original image to the predicted image using ExifTool.

    Args:
    - original_image_path (str): Path to the original image with metadata.
    - predicted_image_path (str): Path to the predicted image where metadata will be copied.
    """
    cmd = [
        "exiftool",
        "-overwrite_original",
        # Copy from original to new
        "-TagsFromFile", original_image_path,
        # Copy all groups of tags (including Makernotes if possible)
        "-all:all",
        # Possibly add other flags if you want to preserve metadata that might otherwise conflict:
        # e.g., "-unsafe" or "-m" in certain tricky cases
        predicted_image_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Copied full metadata from {original_image_path} → {predicted_image_path}")

def generate_jgw_file(styled_path, latitude, longitude, altitude, utm_easting, utm_northing):
    """
    Generate a .jgw world file for the styled JPEG, 
    letting QGIS or other GIS software place the image as a georeferenced raster.
    """
    # Open the styled image to get its pixel dimensions
    img = Image.open(styled_path)
    W, H = img.size

    # If you have UTM coords, you can place the image in meters; if only lat/lon, use degrees
    # For demonstration, let's assume lat/lon degrees/pixel if no altitude info:
    #   each pixel ~ small fraction of a degree
    # If you want more accurate scaling, compute from altitude + camera FOV 
    # or if using UTM, compute meter-based scale.
    
    # Default: small scale in degrees if no altitude or no UTM
    pixel_size_x = 0.00001  # degrees/pixel horizontally
    pixel_size_y = -0.00001 # negative for top-left origin, degrees/pixel vertically
    C = longitude
    F = latitude
    B = 0.0
    D = 0.0

    if utm_easting and utm_northing and altitude and altitude > 0:
        # Example approximate calculation for UTM in meters:
        # FOV-based approach or a simpler approach to define meter/pixel:
        fov_h = math.radians(60.0)  # approximate horizontal FOV
        ground_width = 2 * altitude * math.tan(fov_h/2)  # in meters
        # do the same for ground_height, etc. 
        # for brevity, let's do a direct approach:
        ground_height = ground_width * (H / W)

        # pixel size in X and Y
        pixel_size_x = ground_width / W  # meters/pixel
        pixel_size_y = - (ground_height / H) # negative Y

        # Let’s center the image on the camera’s UTM coordinate 
        # (assuming lat/lon => utm_easting/northing are the photo center):
        x_center = utm_easting
        y_center = utm_northing

        # Top-left pixel center offset
        x_center_pix = (W - 1) / 2.0
        y_center_pix = (H - 1) / 2.0

        A = pixel_size_x
        E = pixel_size_y
        B = 0.0
        D = 0.0

        # The top-left corner (C,F):
        C = x_center - (A * x_center_pix + B * y_center_pix)
        F = y_center - (D * x_center_pix + E * y_center_pix)
    else:
        # If altitude or UTM not available, fallback to lat/lon degrees
        # Let’s place the top-left at (lat,lon), or center, etc.
        # We'll do a simple approach: 
        A = pixel_size_x
        B = 0.0
        D = 0.0
        E = pixel_size_y

        # The top-left corner in lat/lon
        # i.e. if the image center is lat/lon, shift half the image 
        x_center_pix = (W - 1) / 2.0
        y_center_pix = (H - 1) / 2.0
        C = longitude - (A * x_center_pix + B * y_center_pix)
        F = latitude - (D * x_center_pix + E * y_center_pix)

    # Write the .jgw file
    jgw_path = os.path.splitext(styled_path)[0] + ".jgw"
    with open(jgw_path, 'w') as wf:
        wf.write(f"{A:.8f}\n{D:.8f}\n{B:.8f}\n{E:.8f}\n{C:.8f}\n{F:.8f}\n")

    print(f"✅ World file saved: {jgw_path}")




def calculate_fov_area_m2(altitude_m, diagonal_fov_deg=82.1, aspect_ratio=(4, 3)):
    """
    Estimate FOV area (in square meters) from altitude above ground level (AGL),
    assuming the camera is pointing vertically downward.
    
    Parameters:
        altitude_m (float): Altitude in meters above ground.
        diagonal_fov_deg (float): Diagonal field of view of the camera in degrees.
        aspect_ratio (tuple): Aspect ratio of the image (width, height).
    
    Returns:
        float: FOV area in square meters, rounded to 2 decimals.
    """
    if altitude_m <= 0:
        return 0.0

    diagonal_fov_rad = math.radians(diagonal_fov_deg)
    w_ratio = aspect_ratio[0] / math.sqrt(aspect_ratio[0]**2 + aspect_ratio[1]**2)
    h_ratio = aspect_ratio[1] / math.sqrt(aspect_ratio[0]**2 + aspect_ratio[1]**2)
    
    h_fov_rad = 2 * math.atan(math.tan(diagonal_fov_rad / 2) * w_ratio)
    v_fov_rad = 2 * math.atan(math.tan(diagonal_fov_rad / 2) * h_ratio)

    width = 2 * altitude_m * math.tan(h_fov_rad / 2)
    height = 2 * altitude_m * math.tan(v_fov_rad / 2)

    area_m2 = width * height
    return round(area_m2, 2)



def get_terrain_elevation_from_tif(lat, lon, tif_path):
    try:
        with rasterio.open(tif_path) as dem:
            dst_crs = dem.crs
            xs, ys = transform("EPSG:4326", dst_crs, [lon], [lat])
            x, y = xs[0], ys[0]
            row, col = dem.index(x, y)
            elevation = dem.read(1)[row, col]

            # Check for NoData
            nodata = dem.nodata
            if elevation == nodata or elevation < -100 or elevation > 6000:
                print(f"⚠️ No valid elevation at {lat},{lon} → value: {elevation}")
                return None

            return round(float(elevation), 2)
    except Exception as e:
        print(f"❌ DEM lookup failed for {lat},{lon}: {e}")
        return None


