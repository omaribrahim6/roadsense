"""
RoadSense Pipeline Module

This module contains the core processing pipeline components:
- extractor: Video frame extraction and GPS from EXIF
- inference: YOLO model inference
- severity: Severity score calculation
- gps: GPS simulation and GPX matching
- dedup: Detection deduplication
"""

# Frame extraction and EXIF GPS (from extractor.py)
from .extractor import (
    extract_frames,
    extract_metadata,
    extract_gps_from_image,
    gps_to_decimal,
    calculate_distance_meters,
    get_frame_dimensions,
    list_image_files,
    IMAGE_FORMATS,
)

# These imports will work once the files are created:
# from .inference import run_inference, RawDetection
# from .severity import calculate_severity
# from .gps import simulate_gps, match_gpx, interpolate_route
# from .dedup import deduplicate

__all__ = [
    # Extractor
    "extract_frames",
    "extract_metadata",
    "extract_gps_from_image",
    "gps_to_decimal",
    "calculate_distance_meters",
    "get_frame_dimensions",
    "list_image_files",
    "IMAGE_FORMATS",
    # Uncomment as files are created:
    # "run_inference",
    # "RawDetection",
    # "calculate_severity",
    # "simulate_gps",
    # "match_gpx",
    # "interpolate_route",
    # "deduplicate",
]