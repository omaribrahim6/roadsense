import pandas as pd
import piexif
from geopy.geocoders import Nominatim
from PIL import Image
from tabulate import tabulate

from pillow_heif import register_heif_opener
register_heif_opener()

def extract_metadata(file):
    metadata = dict()

    if file.endswith(".heif") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".tiff") or file.endswith(".bmp") or file.endswith(".gif") or file.endswith(
                ".webp") or file.endswith(".psd") or file.endswith(".raw") or file.endswith(".cr2") or file.endswith(".nef") or file.endswith(".heic") or file.endswith(".sr2"):
        try:
            with Image.open(file) as img:
                print("image open")

                exif_data = piexif.load(img.info["exif"])
                
                print("exif data acquired")

                gps_latitude = exif_data['GPS'][piexif.GPSIFD.GPSLatitude]
                gps_latitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLatitudeRef]
                gps_longitude = exif_data['GPS'][piexif.GPSIFD.GPSLongitude]
                gps_longitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLongitudeRef]
                
                gps_latitude_decimal = gps_to_decimal(gps_latitude, gps_latitude_ref)
                gps_longitude_decimal = gps_to_decimal(gps_longitude, gps_longitude_ref)
                
                metadata = {
                    'filename': file,
                    'gps_latitude': gps_latitude_decimal,
                    'gps_longitude': gps_longitude_decimal,
                    'datetime': exif_data['0th'][piexif.ImageIFD.DateTime],
                    }

        except Exception as e:
            print("Error processing image at path {0}: {1}".format(file, repr(e)))

    else:
        print("File at path {} is not an image".format(file))

        
    return metadata

def gps_to_decimal(coord, ref):
    decimal = coord[0][0] / coord[0][1] + coord[1][0] / \
        (60 * coord[1][1]) + coord[2][0] / (3600 * coord[2][1])
    
    if ref in ['S', 'W']:
        decimal *= 1
    
    return decimal

if __name__ == "__main__":
    dir_path = "/media/momen/OS/Users/momen/Work/roadsense/backend/python_modules/HMD_Nokia_8.3_5G.heif"

    metadata = extract_metadata(dir_path)

    print(metadata)
