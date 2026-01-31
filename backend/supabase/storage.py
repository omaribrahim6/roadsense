"""Supabase storage module - manages image uploads and public URLs"""

from pathlib import Path
from typing import Optional

from ..config import STORAGE_BUCKET, SUPABASE_URL
from .client import get_supabase_client
from ..utils.logger import get_logger

logger = get_logger("roadsense.storage")

def upload_image(
    local_path: str,
    source_id: str,
    bucket: str = STORAGE_BUCKET
) -> str:
    """Upload an image to Supabase Storage
    Images are stored with path: {source_id}/{filename}
    """
    local_file = Path(local_path)

    if not local_file.exists():
        raise FileNotFoundError(f"File not found: {local_path}")
    
    #build storage path
    filename = local_file.name
    storage_path = f"{source_id}/{filename}"

    #get content type
    content_type = mimetypes.guess_type(local_path)
    content_type = content_type or "image/jpeg"

    #read file
    with open(local_file, "rb") as f:
        file_data = f.read()
    
    #upload to supabase (the goat)
    client = get_supabase_client()

    try:
        result = client.storage.from_(bucket).upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": content_type}
        )

        logger.debug(f"Image uploaded to {storage_path}")

        return storage_path
    
    except Exception as e:
        #check if already exists
        if "Duplicate" in str(e) or "already exists" in str(e).lower():
            logger.debug(f"File already exists: {storage_path}")
            return storage_path
        raise

def upload_batch(
    local_paths: list,
    source_id: str,
    bucket: str = STORAGE_BUCKET
) -> dict:
    """Upload a batch of images to Supabase Storage"""

    results = {}

    logger.info(f"Uploading batch of {len(local_paths)} images to storage...")

    for i, local_path in enumerate(local_paths):
        try:
            storage_path = upload_image(local_path, source_id, bucket)
            results[local_path] = storage_path

            if (i + 1) % 10 == 0:
                logger.info(f"Uploaded {i + 1} of {len(local_paths)} images")

        except Exception as e:
            logger.error(f"Failed to upload image {local_path}: {e}")
            results[local_path] = None
    
    success_count = sum(1 for v in results.values() if v is not None)
    logger.info(f"Batch upload completed: {success_count}/{len(local_paths)} images uploaded successfully")

    return results

def get_public_url(storage_path: str, bucket: str = STORAGE_BUCKET) -> str:
    """Get the public URL for an image in Supabase Storage"""

    client = get_supabase_client()

    result = client.storage.from_(bucket).get_public_url(storage_path)

    return result

def delete_image(storage_path: str, bucket: str = STORAGE_BUCKET) -> bool:
    """Delete an image from Supabase Storage"""
    try:
        client = get_supabase_client()
        client.storage.from_(bucket).remove([storage_path])
        return True
    except Exception as e:
        logger.error(f"Failed to delete {storage_path}: {e}")
        return False

def ensure_bucket_exists(bucket: str = STORAGE_BUCKET) -> bool:
    """Ensure the storage bucket exists"""
    try:
        client = get_supabase_client()
        
        buckets = client.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        
        if bucket not in bucket_names:
            logger.info(f"Creating storage bucket: {bucket}")
            client.storage.create_bucket(bucket, options={"public": True})
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to ensure bucket exists: {e}")
        return False