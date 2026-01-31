"""Supabase module"""

from .client import get_supabase_client
from .storage import upload_image, get_public_url
from .database import insert_detection, insert_batch_detections

__all__ = [
    "get_supabase_client",
    "upload_image",
    "get_public_url",
    "insert_detection",
    "insert_batch_detections",
]