"""YOLO inference module - runs object detection on frames using HuggingFace model"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(backend_dir / ".env")

try:
    from config import (
        YOLO_MODEL_PATH,
        DEFAULT_CONFIDENCE_THRESHOLD,
        CLASS_MAPPING,
    )
except ImportError:
    YOLO_MODEL_PATH = "cazzz307/Pothole-Finetuned-YoloV8"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    CLASS_MAPPING = {
        "pothole": "pothole",
        "Pothole": "pothole",
    }

try:
    from utils.logger import get_logger
    logger = get_logger("roadsense.inference")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger("roadsense.inference")


# HuggingFace model identifier
HUGGINGFACE_MODEL_ID = "cazzz307/Pothole-Finetuned-YoloV8"


@dataclass
class RawDetection:
    """Raw detection from YOLO model (before GPS enrichment)"""
    frame_path: str
    timestamp: float
    damage_type: str  # Normalized type (pothole, crack, etc.)
    confidence: float
    bbox: Dict[str, int]  # x1, y1, x2, y2
    frame_width: int
    frame_height: int
    original_class: str  # Original class name from model


# Global model instance (lazy loaded)
_model = None


def _download_from_huggingface(repo_id: str, token: Optional[str] = None) -> str:
    """Download YOLO model from HuggingFace Hub
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'cazzz307/Pothole-Finetuned-YoloV8')
        token: HuggingFace API token for gated models
    
    Returns:
        Path to downloaded model file
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
    
    logger.info(f"Downloading model from HuggingFace: {repo_id}")
    
    # Find the model file in the repo (usually best.pt or model.pt)
    try:
        files = list_repo_files(repo_id, token=token)
        model_files = [f for f in files if f.endswith('.pt')]
        
        if not model_files:
            raise FileNotFoundError(f"No .pt model files found in {repo_id}")
        
        # Prefer 'best.pt' if available, otherwise take the first .pt file
        if 'best.pt' in model_files:
            model_file = 'best.pt'
        else:
            model_file = model_files[0]
        
        logger.info(f"Found model file: {model_file}")
        
    except Exception as e:
        logger.warning(f"Could not list repo files: {e}, trying 'best.pt'")
        model_file = 'best.pt'
    
    # Download the model
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_file,
        token=token,
    )
    
    logger.info(f"Model downloaded to: {model_path}")
    return model_path


def _load_model(model_path: Optional[str] = None) -> "YOLO":
    """Load YOLO model from HuggingFace or local path
    
    Args:
        model_path: Path to local .pt file or HuggingFace model ID
                   If None, uses HuggingFace model
    
    Returns:
        Loaded YOLO model
    """
    global _model
    
    if _model is not None:
        return _model
    
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        raise ImportError("ultralytics package required. Install with: pip install ultralytics")
    
    # Determine model source
    if model_path is None:
        model_path = YOLO_MODEL_PATH
    
    # Check if it's a local file
    local_path = Path(model_path)
    if local_path.exists() and local_path.suffix == ".pt":
        logger.info(f"Loading YOLO model from local file: {model_path}")
        _model = YOLO(str(local_path))
    elif "/" in model_path and not local_path.exists():
        # Looks like a HuggingFace model ID (contains /)
        logger.info(f"Detected HuggingFace model ID: {model_path}")
        
        # Get HuggingFace token from environment
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("Using HF_TOKEN from environment for authentication")
        
        try:
            # Download from HuggingFace
            downloaded_path = _download_from_huggingface(model_path, token=hf_token)
            _model = YOLO(downloaded_path)
            logger.info("Model loaded successfully from HuggingFace")
        except Exception as e:
            logger.error(f"Failed to load from HuggingFace: {e}")
            logger.info("Tip: Make sure you have:")
            logger.info("  1. Accepted the model terms at:")
            logger.info(f"     https://huggingface.co/{model_path}")
            logger.info("  2. Set HF_TOKEN in your .env file with a valid HuggingFace token")
            logger.info("  3. The token has read access to gated repos")
            raise
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return _model


def _normalize_class(class_name: str) -> str:
    """Normalize class name to standard damage type"""
    # Check direct mapping
    if class_name in CLASS_MAPPING:
        return CLASS_MAPPING[class_name]
    
    # Check case-insensitive
    lower_name = class_name.lower()
    for key, value in CLASS_MAPPING.items():
        if key.lower() == lower_name:
            return value
    
    # Default to pothole for this model (it's pothole-specific)
    if "pothole" in lower_name or "hole" in lower_name:
        return "pothole"
    
    # Unknown class - log and return as-is
    logger.warning(f"Unknown class name: {class_name}, treating as 'pothole'")
    return "pothole"


def run_inference(
    frame_paths: List[str],
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    timestamps: Optional[List[float]] = None,
    model_path: Optional[str] = None,
    iou_threshold: float = 0.45,
) -> List[RawDetection]:
    """Run YOLO inference on a batch of frames
    
    Args:
        frame_paths: List of paths to frame images
        conf_threshold: Minimum confidence threshold for detections
        timestamps: Optional list of timestamps for each frame
        model_path: Optional path to model (local or HuggingFace ID)
        iou_threshold: IoU threshold for NMS
    
    Returns:
        List of RawDetection objects
    """
    if not frame_paths:
        logger.warning("No frame paths provided for inference")
        return []
    
    # Default timestamps if not provided
    if timestamps is None:
        timestamps = [float(i) for i in range(len(frame_paths))]
    
    # Ensure timestamps match frames
    if len(timestamps) != len(frame_paths):
        logger.warning(f"Timestamp count ({len(timestamps)}) != frame count ({len(frame_paths)})")
        timestamps = [float(i) for i in range(len(frame_paths))]
    
    # Load model
    model = _load_model(model_path)
    
    detections: List[RawDetection] = []
    
    logger.info(f"Running inference on {len(frame_paths)} frames...")
    logger.info(f"  Confidence threshold: {conf_threshold}")
    logger.info(f"  IoU threshold: {iou_threshold}")
    
    # Process frames in batches for efficiency
    batch_size = 16
    total_detections = 0
    
    for batch_start in range(0, len(frame_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(frame_paths))
        batch_paths = frame_paths[batch_start:batch_end]
        batch_timestamps = timestamps[batch_start:batch_end]
        
        try:
            # Run inference on batch
            results = model(
                batch_paths,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
            
            # Process results
            for idx, result in enumerate(results):
                frame_path = batch_paths[idx]
                timestamp = batch_timestamps[idx]
                
                # Get frame dimensions
                frame_height, frame_width = result.orig_shape
                
                # Process boxes
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Get confidence
                        confidence = box.conf[0].item()
                        
                        # Get class name
                        class_id = int(box.cls[0].item())
                        class_names = result.names
                        original_class = class_names.get(class_id, "pothole")
                        
                        # Normalize class to standard damage type
                        damage_type = _normalize_class(original_class)
                        
                        # Create detection
                        detection = RawDetection(
                            frame_path=frame_path,
                            timestamp=timestamp,
                            damage_type=damage_type,
                            confidence=round(confidence, 4),
                            bbox={
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                            },
                            frame_width=frame_width,
                            frame_height=frame_height,
                            original_class=original_class,
                        )
                        
                        detections.append(detection)
                        total_detections += 1
            
            # Progress logging
            if (batch_end) % 50 == 0 or batch_end == len(frame_paths):
                logger.info(f"  Processed {batch_end}/{len(frame_paths)} frames, {total_detections} detections so far")
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
            continue
    
    logger.info(f"Inference complete: {len(detections)} detections from {len(frame_paths)} frames")
    
    return detections


def detect_single_image(
    image_path: str,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_path: Optional[str] = None,
) -> List[Dict]:
    """Convenience function to detect potholes in a single image
    
    Args:
        image_path: Path to image file
        conf_threshold: Minimum confidence threshold
        model_path: Optional path to model
    
    Returns:
        List of detection dicts with keys: class, confidence, bbox, area
    """
    detections = run_inference(
        frame_paths=[image_path],
        conf_threshold=conf_threshold,
        timestamps=[0.0],
        model_path=model_path,
    )
    
    results = []
    for det in detections:
        bbox = det.bbox
        area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
        
        results.append({
            "class": det.damage_type,
            "original_class": det.original_class,
            "confidence": det.confidence,
            "bbox": bbox,
            "area": area,
            "frame_width": det.frame_width,
            "frame_height": det.frame_height,
        })
    
    return results


if __name__ == "__main__":
    # Test inference on a single image
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pothole detection")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    print(f"Testing inference on: {args.image}")
    
    results = detect_single_image(args.image, conf_threshold=args.conf)
    
    if results:
        print(f"\nFound {len(results)} pothole(s):")
        for i, det in enumerate(results, 1):
            print(f"  {i}. {det['class']} (conf: {det['confidence']:.2%})")
            print(f"     Bbox: {det['bbox']}")
            print(f"     Area: {det['area']} pixels")
    else:
        print("\nNo potholes detected.")
