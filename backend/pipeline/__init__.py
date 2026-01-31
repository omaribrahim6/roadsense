"""Pipeline module"""

from .extractor import extract_frames
from .inference import run_inference, RawDetection
from .severity import calculate_severity
from .gps import simulate_gps, match_gpx, interpolate_route
from .dedup import deduplicate

__all__ = [
    "extract_frames",
    "run_inference",
    "RawDetection",
    "calculate_severity",
    "simulate_gps",
    "match_gpx",
    "interpolate_route",
    "deduplicate",
]