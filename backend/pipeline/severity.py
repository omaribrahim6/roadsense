"""Severity calculation module - scores damage severity from 0-1"""

import sys
import logging
from typing import Dict, Optional, List
from pathlib import Path

#add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from config import DAMAGE_TYPE_WEIGHTS
except ImportError:
    DAMAGE_TYPE_WEIGHTS = {
        "pothole": 1.0,
        "crack": 0.6,
        "rut": 0.8,
        "debris": 0.5,
    }

try:
    from utils.logger import get_logger
    logger = get_logger("roadsense.severity")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger("roadsense.severity")

from .inference import RawDetection


def calculate_area_ratio(bbox: Dict[str, int], frame_width: int, frame_height: int) -> float:
    """calculate ratio of bbox area to frame area"""
    bbox_width = bbox["x2"] - bbox["x1"]
    bbox_height = bbox["y2"] - bbox["y1"]
    bbox_area = bbox_width * bbox_height
    
    frame_area = frame_width * frame_height
    
    if frame_area == 0:
        return 0.0
    
    ratio = bbox_area / frame_area
    return min(max(ratio, 0.0), 1.0)


def calculate_severity(
    detection: RawDetection,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None
) -> float:
    """calculate severity score for a detection
    formula: severity = (area_ratio * 0.4) + (type_weight * 0.4) + (confidence * 0.2)
    """
    #use frame dimensions from detection if not provided
    width = frame_width or detection.frame_width
    height = frame_height or detection.frame_height
    
    #1. calculate area ratio (0.0 - 1.0)
    area_ratio = calculate_area_ratio(detection.bbox, width, height)
    
    #normalize area - damages rarely exceed 10% of frame
    normalized_area = min(area_ratio * 10, 1.0)
    
    #2. get damage type weight (0.0 - 1.0)
    type_weight = DAMAGE_TYPE_WEIGHTS.get(detection.damage_type, 0.5)
    
    #3. get confidence score (already 0.0 - 1.0)
    confidence = detection.confidence
    
    #4. calculate weighted severity
    #formula: 40% area + 40% type + 20% confidence
    severity = (normalized_area * 0.4) + (type_weight * 0.4) + (confidence * 0.2)
    
    #clamp to 0-1 range
    severity = min(max(severity, 0.0), 1.0)
    
    return round(severity, 4)


def calculate_severity_batch(detections: List[RawDetection]) -> List[float]:
    """calculate severity for a batch of detections"""
    return [calculate_severity(det) for det in detections]


def get_severity_label(severity: float) -> str:
    """convert numeric severity to label"""
    if severity < 0.25:
        return "low"
    elif severity < 0.5:
        return "medium"
    elif severity < 0.75:
        return "high"
    else:
        return "critical"


def get_severity_color(severity: float) -> str:
    """get hex color for severity visualization"""
    if severity < 0.25:
        return "#22c55e"  #green
    elif severity < 0.5:
        return "#eab308"  #yellow
    elif severity < 0.75:
        return "#f97316"  #orange
    else:
        return "#ef4444"  #red
