"""Supabase module - database and storage operations"""

from .client import get_supabase_client, test_connection, reset_client
from .storage import (
    upload_image,
    upload_batch,
    get_public_url,
    delete_image,
    ensure_bucket_exists,
)
from .database import (
    insert_detection,
    insert_batch_detections,
    get_detections_by_source,
    delete_detection_by_source,
    get_detection_stats,
)

__all__ = [
    #client
    "get_supabase_client",
    "test_connection",
    "reset_client",
    #storage
    "upload_image",
    "upload_batch",
    "get_public_url",
    "delete_image",
    "ensure_bucket_exists",
    #database
    "insert_detection",
    "insert_batch_detections",
    "get_detections_by_source",
    "delete_detection_by_source",
    "get_detection_stats",
]