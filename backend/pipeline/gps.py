"""GPS resolution module - simulates GPS coordinates and matches GPX tracks"""

import sys
import logging
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

#add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from utils.logger import get_logger
    logger = get_logger("roadsense.gps")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger("roadsense.gps")

#optional gpx support
try:
    import gpxpy
    import gpxpy.gpx
    HAS_GPXPY = True
except ImportError:
    HAS_GPXPY = False
    logger.warning("gpxpy not installed, GPX matching disabled")


@dataclass
class GPSPoint:
    """gps coordinate with optional timestamp"""
    lat: float
    lng: float
    timestamp: Optional[float] = None


#demo routes for testing
DEMO_ROUTE_KUWAIT = [
    (29.3759, 47.9774),  #start: kuwait city center
    (29.3780, 47.9800),
    (29.3800, 47.9850),
    (29.3820, 47.9900),
    (29.3850, 47.9950),  #end
]

DEMO_ROUTE_SF = [
    (37.7749, -122.4194),  #start: sf downtown
    (37.7751, -122.4180),
    (37.7755, -122.4160),
    (37.7760, -122.4140),
    (37.7770, -122.4120),  #end
]


def interpolate_route(
    timestamp: float,
    route: List[Tuple[float, float]],
    total_duration: float
) -> Tuple[float, float]:
    """interpolate gps position along a route based on timestamp"""
    if not route:
        raise ValueError("Route cannot be empty")
    
    if len(route) == 1:
        return route[0]
    
    #clamp timestamp to valid range
    timestamp = max(0.0, min(timestamp, total_duration))
    
    #calculate progress along route (0.0 to 1.0)
    progress = timestamp / total_duration if total_duration > 0 else 0.0
    
    #calculate which segment we're on
    num_segments = len(route) - 1
    segment_progress = progress * num_segments
    segment_index = int(segment_progress)
    
    #handle edge case at end of route
    if segment_index >= num_segments:
        return route[-1]
    
    #interpolation factor within segment (0.0 to 1.0)
    t = segment_progress - segment_index
    
    #get segment endpoints
    start = route[segment_index]
    end = route[segment_index + 1]
    
    #linear interpolation
    lat = start[0] + (end[0] - start[0]) * t
    lng = start[1] + (end[1] - start[1]) * t
    
    return (round(lat, 6), round(lng, 6))


def simulate_gps(
    timestamp: float,
    route: Optional[List[Tuple[float, float]]] = None,
    total_duration: float = 60.0
) -> Tuple[float, float]:
    """simulate gps coordinates based on timestamp and route"""
    if route is None:
        route = DEMO_ROUTE_KUWAIT
    
    return interpolate_route(timestamp, route, total_duration)


def match_gpx(
    timestamp: float,
    gpx_path: str,
    video_start_offset: float = 0.0
) -> Tuple[float, float]:
    """match timestamp to closest gpx track point"""
    if not HAS_GPXPY:
        raise ImportError("gpxpy is required for GPX matching")
    
    gpx_file = Path(gpx_path)
    if not gpx_file.exists():
        raise FileNotFoundError(f"GPX file not found: {gpx_path}")
    
    with open(gpx_file, "r") as f:
        gpx = gpxpy.parse(f)
    
    #collect all track points with their timestamps
    points: List[GPSPoint] = []
    start_time = None
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is None:
                    continue
                
                if start_time is None:
                    start_time = point.time
                
                #calculate seconds from start
                delta = (point.time - start_time).total_seconds()
                points.append(GPSPoint(
                    lat=point.latitude,
                    lng=point.longitude,
                    timestamp=delta
                ))
    
    if not points:
        raise ValueError("No valid GPS points found in GPX file")
    
    #adjust timestamp with offset
    adjusted_timestamp = timestamp + video_start_offset
    
    #find closest point
    closest = min(points, key=lambda p: abs(p.timestamp - adjusted_timestamp))
    
    return (round(closest.lat, 6), round(closest.lng, 6))


def parse_gpx_route(gpx_path: str) -> Tuple[List[Tuple[float, float]], float]:
    """parse gpx file into route and total duration"""
    if not HAS_GPXPY:
        raise ImportError("gpxpy is required for GPX parsing")
    
    with open(gpx_path, "r") as f:
        gpx = gpxpy.parse(f)
    
    route = []
    start_time = None
    end_time = None
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                route.append((point.latitude, point.longitude))
                
                if point.time:
                    if start_time is None:
                        start_time = point.time
                    end_time = point.time
    
    #calculate duration
    if start_time and end_time:
        total_duration = (end_time - start_time).total_seconds()
    else:
        #estimate based on typical speed
        total_duration = len(route) * 1.0
    
    return route, total_duration
