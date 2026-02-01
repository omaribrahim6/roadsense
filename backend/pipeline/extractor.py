"""
Frame Extractor Module

Extracts frames from video files and metadata from images.
Handles GPS extraction from EXIF data for geotagged photos.
Also includes GPS distance calculation utilities.
"""

import cv2
import sys
import math
import logging
import piexif
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

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

# Supported image formats
IMAGE_FORMATS = ('.jpg', '.jpeg', '.gif', '.png', '.bmp', '.tiff', '.raw', '.cr2', '.heic', '.heif', '.webp')


def extract_metadata(file_path: str) -> Dict:
    """
    Extract metadata including GPS coordinates from an image file.
    
    Args:
        file_path: Path to the image file
    
    Returns:
        Dictionary with filename, gps_latitude, gps_longitude, datetime
        Empty dict if extraction fails
    """
    metadata = {}

    if not file_path.lower().endswith(IMAGE_FORMATS):
        logger.warning(f"File is not a supported image format: {file_path}")
        return metadata

    try:
        with Image.open(file_path) as img:
            if "exif" not in img.info:
                logger.debug(f"No EXIF data in image: {file_path}")
                return metadata

            exif_data = piexif.load(img.info["exif"])
            
            # Check if GPS data exists
            if 'GPS' not in exif_data or not exif_data['GPS']:
                logger.debug(f"No GPS data in EXIF: {file_path}")
                return metadata

            gps_data = exif_data['GPS']
            
            # Check for required GPS fields
            if piexif.GPSIFD.GPSLatitude not in gps_data:
                logger.debug(f"No GPS latitude in EXIF: {file_path}")
                return metadata

            gps_latitude = gps_data[piexif.GPSIFD.GPSLatitude]
            gps_latitude_ref = gps_data.get(piexif.GPSIFD.GPSLatitudeRef, b'N')
            gps_longitude = gps_data[piexif.GPSIFD.GPSLongitude]
            gps_longitude_ref = gps_data.get(piexif.GPSIFD.GPSLongitudeRef, b'E')
            
            gps_latitude_decimal = gps_to_decimal(gps_latitude, gps_latitude_ref)
            gps_longitude_decimal = gps_to_decimal(gps_longitude, gps_longitude_ref)
            
            # Get datetime if available
            datetime_val = None
            if '0th' in exif_data and piexif.ImageIFD.DateTime in exif_data['0th']:
                datetime_val = exif_data['0th'][piexif.ImageIFD.DateTime]
                if isinstance(datetime_val, bytes):
                    datetime_val = datetime_val.decode('utf-8')
            
            metadata = {
                'filename': file_path,
                'gps_latitude': gps_latitude_decimal,
                'gps_longitude': gps_longitude_decimal,
                'datetime': datetime_val,
            }
            
            logger.debug(f"Extracted GPS from {file_path}: ({gps_latitude_decimal}, {gps_longitude_decimal})")

    except Exception as e:
        logger.debug(f"Error extracting metadata from {file_path}: {repr(e)}")

    return metadata


def gps_to_decimal(coord: tuple, ref) -> float:
    """
    Convert GPS coordinates from degrees/minutes/seconds to decimal degrees.
    
    Args:
        coord: Tuple of ((deg_num, deg_den), (min_num, min_den), (sec_num, sec_den))
        ref: Reference direction ('N', 'S', 'E', 'W') as string or bytes
    
    Returns:
        Decimal degrees (negative for S/W)
    """
    degrees = coord[0][0] / coord[0][1]
    minutes = coord[1][0] / coord[1][1]
    seconds = coord[2][0] / coord[2][1]
    
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    
    # Handle both string and bytes reference
    if ref in ['S', 'W', b'S', b'W']:
        decimal *= -1
    
    return round(decimal, 6)


def calculate_distance_meters(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    Calculate distance between two GPS points using the Haversine formula.
    
    Args:
        point1: (latitude, longitude) tuple
        point2: (latitude, longitude) tuple
    
    Returns:
        Distance in meters
    """
    try:
        from haversine import haversine, Unit
        return haversine(point1, point2, unit=Unit.METERS)
    except ImportError:
        # Fallback implementation using Haversine formula
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in meters
        r = 6371000
        
        return c * r


def extract_gps_from_image(image_path: str) -> Optional[Tuple[float, float]]:
    """
    Extract GPS coordinates from an image file.
    Convenience function that returns just the coordinates.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Tuple of (latitude, longitude) or None if no GPS data
    """
    metadata = extract_metadata(image_path)
    
    if metadata and 'gps_latitude' in metadata and 'gps_longitude' in metadata:
        return (metadata['gps_latitude'], metadata['gps_longitude'])
    
    return None


def get_frame_dimensions(frame_path: str) -> Tuple[int, int]:
    """
    Get the dimensions of a frame image.
    
    Args:
        frame_path: Path to the frame image
    
    Returns:
        Tuple of (width, height)
    
    Raises:
        ValueError: If image cannot be read
    """
    img = cv2.imread(frame_path)
    if img is None:
        raise ValueError(f"Cannot read image: {frame_path}")
    
    height, width = img.shape[:2]
    return width, height


def list_image_files(
    directory: str,
    extensions: Tuple[str, ...] = IMAGE_FORMATS
) -> List[str]:
    """
    List all image files in a directory.
    
    Args:
        directory: Directory path to scan
        extensions: Tuple of valid file extensions
    
    Returns:
        List of image file paths sorted by name
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    image_files = []
    for ext in extensions:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))
    
    # Sort by filename
    image_files = sorted(set(image_files), key=lambda p: p.name)
    
    return [str(f) for f in image_files]

def extract_frames(
    video_path: str,
    output_path: str,
    fps: int = 1,
    max_frames: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Extract frames from a video file at the specified FPS.
    
    Args:
        video_path: Path to the input video file
        output_path: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)
        max_frames: Maximum number of frames to extract (None = all)
    
    Returns:
        List of tuples: (frame_path, timestamp_seconds)
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened
    """
    video_dir = Path(video_path)
    output_dir = Path(output_path)
    extracted_frames: List[Tuple[str, float]] = []

    try:
        if not video_dir.exists():
            raise FileNotFoundError(f"Video file not found: {video_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_dir))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_dir}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        logger.info(f"Video: {video_dir.name}")
        logger.info(f"  FPS: {video_fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        logger.info(f"  Extracting at {fps} fps...")

        # Calculate frame interval
        frame_interval = int(video_fps / fps) if fps < video_fps else 1

        frame_count = 0
        extracted_count = 0

        while True:
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
        logger.error(f"Error processing video at path {video_dir}: {repr(e)}")
    
    return extracted_frames

if __name__ == "__main__":
    # Test image metadata extraction
    # image_path = "path/to/test/image.heif"
    # metadata = extract_metadata(image_path)
    # print(f"Metadata: {metadata}")
    
    # gps = extract_gps_from_image(image_path)
    # print(f"GPS: {gps}")
    
    # Test video frame extraction
    video_path = r"C:\Users\momen\Repos\roadsense\backend\pipeline\Neon Dingo Demo.mp4"
    output_path = r"C:\Users\momen\Repos\roadsense\backend\output"
    
    frames = extract_frames(video_path, output_path, fps=1)
    print(f"Extracted {len(frames)} frames")
    
    if frames:
        print(f"First frame: {frames[0]}")
        print(f"Last frame: {frames[-1]}")
