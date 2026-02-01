import cv2
import piexif
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from utils.logger import get_logger
    logger = get_logger("roadsense.extractor")
except ImportError:
    # Fallback if utils not available
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger("roadsense.extractor")

from pillow_heif import register_heif_opener

register_heif_opener()

image_formats = ['.jpg', '.gif', '.jpeg', '.png', '.bmp', '.tiff', '.raw', '.cr2', '.heic', '.webp']

def extract_metadata(file):
    metadata = dict()

    if file.endswith(tuple(image_formats)):
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
    
    if ref in ['S', 'W', b'S', b'W']:
        decimal *= -1
    
    return decimal

def extract_frames(video_path, output_path, fps=1, max_frames=None):
    
    video_dir = Path(video_path)
    output_dir = Path(output_path)

    try:
        if not video_dir.exists():
            raise FileNotFoundError(f"Video file not found: {video_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_dir))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_dir}")

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0

            logger.info(f"Video: {video_dir.name}")
            logger.info(f"  FPS: {video_fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")
            logger.info(f"  Extracting at {fps} fps...")

            # Calculate frame interval
            frame_interval = int(video_fps / fps) if fps < video_fps else 1

            extracted_frames: List[Tuple[str, float]] = []
            frame_count = 0
            extracted_count = 0

            while cap.isOpened():
                success, frame = cap.read()
                
                if not success:
                    break

                if frame_count % frame_interval == 0:
                
                    # Calculate timestamp
                    timestamp_seconds = frame_count / video_fps
                    
                    # Generate frame filename
                    frame_filename = f"frame_{extracted_count:05d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    # Save frame as JPG
                    cv2.imwrite(str(frame_path), frame)
                    
                    extracted_frames.append((str(frame_path), timestamp_seconds))
                    extracted_count += 1
                    
                    if extracted_count % 10 == 0:
                        
                        logger.info(f"  Extracted {extracted_count} frames...")
                        

                    # Check max frames limit
                    if max_frames and extracted_count >= max_frames:
                        
                        logger.info(f"  Reached max frames limit: {max_frames}")
                        
                        break
            
                frame_count += 1

            cap.release()

            logger.info(f"  Extraction complete: {extracted_count} frames")

    except Exception as e:
        print("Error processing video at path {0}: {1}".format(video_dir, repr(e)))
    
    return extracted_frames

if __name__ == "__main__":
    
    dir_path = "/media/momen/OS/Users/momen/Work/roadsense/backend/pipeline/HMD_Nokia_8.3_5G.heif"

    # metadata = extract_metadata(dir_path)

    # print(metadata)
    
    video_path = r"C:\Users\momen\Repos\roadsense\backend\pipeline\Neon Dingo Demo.mp4"

    extract_frames(video_path, r"C:\Users\momen\Repos\roadsense\backend\output")
