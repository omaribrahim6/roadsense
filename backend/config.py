"""Configuration for the RoadSense backend"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "weights"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(MODELS_DIR / "best.pt"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.25"))

DEFAULT_FPS = int(os.getenv("DEFAULT_FPS", "1"))
DEDUP_RADIUS_METERS = 10.0

STORAGE_BUCKET = "detections-images"

DAMAGE_TYPE_WEIGHTS = {
    "pothole": 1.0,
    "crack": 0.6,
    "rut": 0.8,
    "debris": 0.5
}

CLASS_MAPPING = {
    "pothole": "pothole",
    "Pothole": "pothole",
    "D00": "crack",
    "D10": "crack",
    "D20": "crack",
    "D30": "crack",
    "D40": "crack",
    "crack": "crack",
    "Crack": "crack",
    "rut": "rut",
    "Rut": "rut",
    "debris": "debris",
    "Debris": "debris",
}