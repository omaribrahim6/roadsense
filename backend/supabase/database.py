"""Supabase database module - manages detection data insertion"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import json

from .client import get_supabase_client
from ..pipeline.dedup import EnrichedDetection
from ..utils.logger import get_logger

logger = get_logger("roadsense.database")

def insert_detection(
    detection: EnrichedDetection,
    source_id: str,
    image_path: str,
    captured_at: Optional[str] = None
) -> Optional[str]:
    """Insert a detection into the Supabase database"""
    client = get_supabase_client()
    
    # build row data
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).isoformat()
    
    row = {
        "captured_at": captured_at,
        "lat": detection.lat,
        "lng": detection.lng,
        "damage_type": detection.damage_type,
        "confidence": detection.confidence,
        "severity": detection.severity,
        "bbox": detection.bbox,  #json
        "image_path": image_path,
        "source_id": source_id,
        "status": "new",
    }
    
    try:
        result = client.table("detections").insert(row).execute()
        
        if result.data and len(result.data) > 0:
            detection_id = result.data[0].get("id")
            logger.debug(f"Inserted detection: {detection_id}")
            return detection_id
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to insert detection: {e}")
        return None

def insert_batch_detections(
    detections: List[EnrichedDetection],
    source_id: str,
    image_paths: Dict[str, str],
    captured_at_base: Optional[datetime] = None
) -> List[str]:
    """Insert a batch of detections into the Supabase database"""
    if not detections:
        return []
    
    client = get_supabase_client()
    
    if captured_at_base is None:
        captured_at_base = datetime.now(timezone.utc)
    
    # build all rows
    rows = []
    for det in detections:
        # calculate capture time based on timestamp offset
        captured_at = captured_at_base.isoformat()
        
        # get storage path for this frame
        storage_path = image_paths.get(det.frame_path, det.frame_path)
        
        row = {
            "captured_at": captured_at,
            "lat": det.lat,
            "lng": det.lng,
            "damage_type": det.damage_type,
            "confidence": det.confidence,
            "severity": det.severity,
            "bbox": det.bbox,
            "image_path": storage_path,
            "source_id": source_id,
            "status": "new",
        }
        rows.append(row)
    
    logger.info(f"Inserting {len(rows)} detections to database...")
    
    try:
        # supabase supports batch insert
        result = client.table("detections").insert(rows).execute()
        
        inserted_ids = [r.get("id") for r in result.data] if result.data else []
        
        logger.info(f"Inserted {len(inserted_ids)} detections")
        
        return inserted_ids
        
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        
        # fall back to individual inserts
        logger.info("Falling back to individual inserts...")
        inserted_ids = []
        
        for det, row in zip(detections, rows):
            try:
                result = client.table("detections").insert(row).execute()
                if result.data:
                    inserted_ids.append(result.data[0].get("id"))
            except Exception as e2:
                logger.error(f"Individual insert failed: {e2}")
        
        return inserted_ids

def get_detections_by_source(source_id: str) -> List[Dict]:
    """Get detections by source ID"""
    client = get_supabase_client()
    
    result = client.table("detections")\
        .select("*")\
        .eq("source_id", source_id)\
        .order("created_at", desc=True)\
        .execute()
    
    return result.data or []

def delete_detection_by_source(source_id: str) -> bool:
    """Delete detections by source ID"""
    client = get_supabase_client()
    
    try:
        result = client.table("detections")\
            .delete()\
            .eq("source_id", source_id)\
            .execute()
        
        count = len(result.data) if result.data else 0
        logger.info(f"Deleted {count} detections for source: {source_id}")
        return count
        
    except Exception as e:
        logger.error(f"Failed to delete detections: {e}")
        return 0

def get_detection_stats(source_id: Optional[str] = None) -> Dict:
    """Get detection statistics"""
    client = get_supabase_client()
    
    query = client.table("detections").select("*")
    
    if source_id:
        query = query.eq("source_id", source_id)
    
    result = query.execute()
    detections = result.data or []
    
    if not detections:
        return {
            "total": 0,
            "by_type": {},
            "avg_confidence": 0,
            "avg_severity": 0,
        }
    
    # calculate stats
    by_type = {}
    total_confidence = 0
    total_severity = 0
    
    for det in detections:
        damage_type = det.get("damage_type", "unknown")
        by_type[damage_type] = by_type.get(damage_type, 0) + 1
        total_confidence += det.get("confidence", 0)
        total_severity += det.get("severity", 0)
    
    return {
        "total": len(detections),
        "by_type": by_type,
        "avg_confidence": round(total_confidence / len(detections), 3),
        "avg_severity": round(total_severity / len(detections), 3),
    }