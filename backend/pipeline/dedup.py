"""Deduplication module - deduplicates detections based on spatial and temporal criteria"""

import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from config import DEDUP_RADIUS_METERS
except ImportError:
    DEDUP_RADIUS_METERS = 10.0  # Default fallback

from .extractor import calculate_distance_meters

try:
    from utils.logger import get_logger
    logger = get_logger("roadsense.dedup")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger("roadsense.dedup")

@dataclass
class EnrichedDetection:
    """Enriched detection with GPS coordinates and severity (final format before database insertion)
    """
    frame_path: str
    timestamp: float
    damage_type: str
    confidence: float
    severity: float
    bbox: Dict[str, int]
    lat: float
    lng: float
    frame_width: int
    frame_height: int
    original_class: str = ""
    frequency: int = 1 # how many times it was detected

    def to_dict(self) -> Dict:
        """Convert to dict for JSON/db"""
        return {
            "frame_path": self.frame_path,
            "timestamp": self.timestamp,
            "damage_type": self.damage_type,
            "confidence": self.confidence,
            "severity": self.severity,
            "bbox": self.bbox,
            "lat": self.lat,
            "lng": self.lng,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "original_class": self.original_class,
            "frequency": self.frequency,
        }

def deduplicate(
    detections: List[EnrichedDetection],
    radius_meters: float = DEDUP_RADIUS_METERS
) -> List[EnrichedDetection]:
    """Deduplicate detections based on spatial proximity and damage type
    group detections by damage type, cluster detections within radius for each type and track frequency
    """
    if not detections:
        return []

    logger.info(f"Deduplicating {len(detections)} detections (radius: {radius_meters}m)")

    # group by damage type
    by_type: Dict[str, List[EnrichedDetection]] = defaultdict(list)
    for det in detections:
        by_type[det.damage_type].append(det)
    
    deduplicated: List[EnrichedDetection] = []
    
    for damage_type, type_detections in by_type.items():
        # cluster detections for this type
        clusters = _cluster_by_location(type_detections, radius_meters)
        
        # merge each cluster
        for cluster in clusters:
            merged = _merge_cluster(cluster)
            deduplicated.append(merged)
        
        logger.info(
            f"  {damage_type}: {len(type_detections)} -> {len(clusters)} unique"
        )
    
    # sort by severity (highest first)
    deduplicated.sort(key=lambda d: d.severity, reverse=True)
    
    logger.info(f"Deduplication complete: {len(detections)} -> {len(deduplicated)}")
    
    return deduplicated

def _cluster_by_location(
    detections: List[EnrichedDetection],
    radius_meters: float
) -> List[List[EnrichedDetection]]:
    """Cluster detections by proximity
    simple approach:
    - start with first detection as cluster center
    - add detections to cluster if within radius
    - repeat for each detection until all are processed
    """
    if not detections:
        return []
    
    # track which detections have been clustered
    unclustered = list(detections)
    clusters: List[List[EnrichedDetection]] = []
    
    while unclustered:
        # start new cluster with first unclustered detection
        current = unclustered.pop(0)
        cluster = [current]
        cluster_center = (current.lat, current.lng)
        
        # find all detections within radius
        remaining = []
        for det in unclustered:
            det_point = (det.lat, det.lng)
            distance = calculate_distance_meters(cluster_center, det_point)
            
            if distance <= radius_meters:
                cluster.append(det)
            else:
                remaining.append(det)
        
        unclustered = remaining
        clusters.append(cluster)
    
    return clusters

def _merge_cluster(cluster: List[EnrichedDetection]) -> EnrichedDetection:
    """Merge a cluster of detections into a single detection
    strat:
    - use detection with highest confidence as the representative
    - keep that detection's image and bbox
    - average the GPS coordinates
    - sum frequency and take max severity
    """
    if len(cluster) == 1:
        return cluster[0]
    
    # find detection with highest confidence
    best = max(cluster, key=lambda d: d.confidence)
    
    # calculate average GPS (weighted by confidence)
    total_weight = sum(d.confidence for d in cluster)
    avg_lat = sum(d.lat * d.confidence for d in cluster) / total_weight
    avg_lng = sum(d.lng * d.confidence for d in cluster) / total_weight
    
    # take max severity
    max_severity = max(d.severity for d in cluster)
    
    # create merged detection
    merged = EnrichedDetection(
        frame_path=best.frame_path,
        timestamp=best.timestamp,
        damage_type=best.damage_type,
        confidence=best.confidence,
        severity=max_severity,
        bbox=best.bbox,
        lat=round(avg_lat, 6),
        lng=round(avg_lng, 6),
        frame_width=best.frame_width,
        frame_height=best.frame_height,
        original_class=best.original_class,
        frequency=len(cluster),
    )
    
    return merged

def filter_by_confidence(
    detections: List[EnrichedDetection],
    min_confidence: float = 0.5
) -> List[EnrichedDetection]:
    """Filter detections by minimum confidence"""
    filtered = [d for d in detections if d.confidence >= min_confidence]
    logger.info(f"Filtered by confidence >= {min_confidence}: {len(detections)} -> {len(filtered)}")
    return filtered

def filter_by_severity(
    detections: List[EnrichedDetection],
    min_severity: float = 0.3
) -> List[EnrichedDetection]:
    """Filter detections by minimum severity"""
    filtered = [d for d in detections if d.severity >= min_severity]
    logger.info(f"Filtered by severity >= {min_severity}: {len(detections)} -> {len(filtered)}")
    return filtered