"""Supabase client module - manages Supabase connections and API interactions"""

import sys
from pathlib import Path
from typing import Optional

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from utils.logger import get_logger

logger = get_logger("roadsense.supabase")

#Global Supabase client instance
_client: Optional[Client] = None

def get_supabase_client() -> Client:
    """Get the Supabase client instance
    Returns ValueError if client creation fails
    """
    global _client

    if _client is not None:
        return _client

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Supabase creds not set, set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
    
    logger.info(f"Connecting to Supabase: {SUPABASE_URL[:50]}...")

    _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    logger.info("Supabase client created successfully")

    return _client

def test_connection() -> bool:
    """Test the Supabase connection"""
    try:
        client = get_supabase_client()

        result = client.table("detections").select("id").limit(1).execute()
        
        logger.info("Supabase connection test successful")
        return True

    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return False

def reset_client() -> None:
    """Reset the Supabase client instance"""
    global _client
    _client = None