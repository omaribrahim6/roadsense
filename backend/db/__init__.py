"""Supabase module - database and storage operations"""

import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from db.client import get_supabase_client, test_connection, reset_client
from db.storage import (
    upload_image,
    upload_batch,
    get_public_url,
    delete_image,
    ensure_bucket_exists,
)
from db.database import (
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