"""RoadSense pipeline module - core processing components"""

#extractor - frame extraction and exif gps
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

#inference - yolo detection
from .inference import run_inference, RawDetection, InferenceEngine

#severity - damage severity scoring
from .severity import (
    calculate_severity,
    calculate_severity_batch,
    get_severity_label,
    get_severity_color,
)

#gps - gps simulation and gpx matching
from .gps import (
    simulate_gps,
    match_gpx,
    interpolate_route,
    parse_gpx_route,
    DEMO_ROUTE_KUWAIT,
    DEMO_ROUTE_SF,
)

#dedup - detection deduplication
from .dedup import (
    deduplicate,
    EnrichedDetection,
    filter_by_confidence,
    filter_by_severity,
)

__all__ = [
    #extractor
    "extract_frames",
    "extract_metadata",
    "extract_gps_from_image",
    "gps_to_decimal",
    "calculate_distance_meters",
    "get_frame_dimensions",
    "list_image_files",
    "IMAGE_FORMATS",
    #inference
    "run_inference",
    "RawDetection",
    "InferenceEngine",
    #severity
    "calculate_severity",
    "calculate_severity_batch",
    "get_severity_label",
    "get_severity_color",
    #gps
    "simulate_gps",
    "match_gpx",
    "interpolate_route",
    "parse_gpx_route",
    "DEMO_ROUTE_KUWAIT",
    "DEMO_ROUTE_SF",
    #dedup
    "deduplicate",
    "EnrichedDetection",
    "filter_by_confidence",
    "filter_by_severity",
]