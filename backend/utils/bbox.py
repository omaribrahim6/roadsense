"""Bounding box utilities"""

from typing import Dict, Tuple

def bbox_to_dict(x1: int, y1: int, x2: int, y2: int) -> Dict[str, int]:
    """Convert bounding box coordinates to dictionary

    Args:
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
    
    Returns:
        Dictionary with x1, y1, x2, y2 keys
    """

    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }

def calculate_area_ratio(
    bbox: Dict[str, int],
    frame_width: int,
    frame_height: int
) -> float:

    """
    Calculate the ratio of bounding box area to frame area.
    
    Args:
        bbox: Bounding box dictionary with x1, y1, x2, y2
        frame_width: Width of the frame in pixels
        frame_height: Height of the frame in pixels
    
    Returns:
        Area ratio between 0.0 and 1.0
    """
    bbox_width = bbox["x2"] - bbox["x1"]
    bbox_height = bbox["y2"] - bbox["y1"]
    bbox_area = bbox_width * bbox_height
    
    frame_area = frame_width * frame_height
    
    if frame_area == 0:
        return 0.0
    
    # Clamp to 0-1 range
    ratio = bbox_area / frame_area
    return min(max(ratio, 0.0), 1.0)


def get_bbox_center(bbox: Dict[str, int]) -> Tuple[int, int]:
    """
    Get the center point of a bounding box.
    
    Args:
        bbox: Bounding box dictionary
    
    Returns:
        Tuple of (center_x, center_y)
    """
    center_x = (bbox["x1"] + bbox["x2"]) // 2
    center_y = (bbox["y1"] + bbox["y2"]) // 2
    return center_x, center_y