# This module handles YOLO (You Only Look Once) inference for object detection.
# It utilizes a model from HuggingFace to perform detection on video frames.

import os  # Importing the os module for operating system dependent functionality.
import sys  # Importing the sys module to manipulate the Python runtime environment.
import logging  # Importing the logging module for logging messages.
from dataclasses import dataclass  # Importing dataclass for creating classes that are primarily used to store data.
from typing import List, Dict, Optional, Tuple  # Importing type hints for better code clarity.
from pathlib import Path  # Importing Path for handling filesystem paths.

import cv2  # Importing OpenCV for image and video processing.
import numpy as np  # Importing NumPy for numerical operations on arrays.

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent  # Getting the parent directory of the current file.
if str(backend_dir) not in sys.path:  # Checking if the backend directory is already in the system path.
    sys.path.insert(0, str(backend_dir))  # Adding backend directory to system path if not present.

# Load environment variables from .env file
from dotenv import load_dotenv  # Importing load_dotenv to load environment variables from a .env file.
load_dotenv(backend_dir / ".env")  # Loading environment variables from the .env file located in the backend directory.

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

# HuggingFace model identifier (https://huggingface.co/cazzz307/Pothole-Finetuned-YoloV8)
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
    
    This function downloads a pre-trained YOLO model from HuggingFace's model hub.
    It automatically discovers .pt (PyTorch) model files in the repository, preferring
    'best.pt' if available. Downloaded files are cached locally for future use.
    
    Args:
        repo_id: HuggingFace repository ID (format: 'username/repo-name')
                Example: 'cazzz307/Pothole-Finetuned-YoloV8'
        token: HuggingFace API token for accessing gated/private models
               Can be obtained from https://huggingface.co/settings/tokens
    
    Returns:
        Local file path to the downloaded model file
    
    Raises:
        ImportError: If huggingface_hub package is not installed
        FileNotFoundError: If no .pt model files are found in the repository
    """
    # Import required HuggingFace utilities for downloading from hub
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
    
    # Log download initiation
    logger.info(f"Downloading model from HuggingFace: {repo_id}")
    
    # Discover available .pt model files in the repository
    try:
        # List all files in the HuggingFace repository
        files = list_repo_files(repo_id, token=token)
        # Filter to find only PyTorch model files (.pt)
        model_files = [f for f in files if f.endswith('.pt')]
        
        # Validate that at least one model file exists
        if not model_files:
            raise FileNotFoundError(f"No .pt model files found in {repo_id}")
        
        # Prefer 'best.pt' (commonly the best checkpoint) if available, otherwise use first found
        if 'best.pt' in model_files:
            model_file = 'best.pt'
        else:
            model_file = model_files[0]
        
        logger.info(f"Found model file: {model_file}")
        
    except Exception as e:
        # If file listing fails, attempt to download 'best.pt' as a fallback
        logger.warning(f"Could not list repo files: {e}, trying 'best.pt'")
        model_file = 'best.pt'
    
    # Download the model file from HuggingFace hub
    # Files are cached in ~/.cache/huggingface/hub/ by default
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_file,
        token=token,  # Authentication token for gated models
    )
    
    logger.info(f"Model downloaded to: {model_path}")
    return model_path

def _load_model(model_path: Optional[str] = None) -> "YOLO":
    """Load YOLO model from HuggingFace or local path
    
    This function loads a YOLO model with intelligent source detection. It caches
    the model globally to avoid redundant loading on subsequent calls. Supports both
    local .pt files and HuggingFace model repository IDs.
    
    Args:
        model_path: Path to local .pt file or HuggingFace model ID (format: 'user/repo')
                   If None, uses YOLO_MODEL_PATH from config
    
    Returns:
        Loaded YOLO model instance ready for inference
    
    Raises:
        ImportError: If ultralytics package not installed or HuggingFace auth fails
        FileNotFoundError: If local model file doesn't exist or model cannot be found
    """
    global _model
    
    # Return cached model if already loaded (avoid redundant loading)
    if _model is not None:
        return _model
    
    # Import YOLO from ultralytics library
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        raise ImportError("ultralytics package required. Install with: pip install ultralytics")
    
    # Use default model path from config if not specified
    if model_path is None:
        model_path = YOLO_MODEL_PATH
    
    # Determine if model_path is a local file or HuggingFace repository ID
    local_path = Path(model_path)
    
    # Case 1: Local model file exists and is a .pt PyTorch model
    if local_path.exists() and local_path.suffix == ".pt":
        logger.info(f"Loading YOLO model from local file: {model_path}")
        _model = YOLO(str(local_path))
        
    # Case 2: HuggingFace model ID detected (contains forward slash and file doesn't exist locally)
    elif "/" in model_path and not local_path.exists():
        logger.info(f"Detected HuggingFace model ID: {model_path}")
        
        # Retrieve HuggingFace authentication token from environment variables
        # This is needed for accessing gated/private models
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("Using HF_TOKEN from environment for authentication")
        
        try:
            # Download model from HuggingFace hub and get local path
            downloaded_path = _download_from_huggingface(model_path, token=hf_token)
            # Load the downloaded YOLO model
            _model = YOLO(downloaded_path)
            logger.info("Model loaded successfully from HuggingFace")
            
        except Exception as e:
            # Provide detailed error information for troubleshooting HuggingFace issues
            logger.error(f"Failed to load from HuggingFace: {e}")
            logger.info("Tip: Make sure you have:")
            logger.info("  1. Accepted the model terms at:")
            logger.info(f"     https://huggingface.co/{model_path}")
            logger.info("  2. Set HF_TOKEN in your .env file with a valid HuggingFace token")
            logger.info("  3. The token has read access to gated repos")
            raise
    else:
        # Neither local file nor valid HuggingFace ID
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return _model


# Color mapping for different damage types (BGR format for OpenCV)
DAMAGE_COLORS = {
    "pothole": (0, 0, 255),      # Red
    "crack": (0, 165, 255),      # Orange
    "rut": (0, 255, 255),        # Yellow
    "debris": (255, 0, 255),     # Magenta
}

DEFAULT_BOX_COLOR = (0, 255, 0)  # Green for unknown types


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict],
    thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw bounding boxes on an image
    
    This function visualizes detection results by drawing colored rectangles around detected
    potholes and other damage, with labels showing the damage type and confidence score.
    Each damage type gets a specific color for easy visual distinction.
    
    Args:
        image: Input image as numpy array (BGR color format, as used by OpenCV)
        detections: List of detection dictionaries with keys:
                   'bbox': dict with x1, y1, x2, y2 coordinates
                   'damage_type' or 'class': type of damage detected
                   'confidence': detection confidence score (0.0-1.0)
        thickness: Line thickness for drawing boxes (in pixels)
        font_scale: Font scale for label text (relative to base font size)
        show_confidence: Whether to display confidence percentage in label
    
    Returns:
        Image with bounding boxes and labels drawn (BGR format)
    """
    # Create a copy to avoid modifying original image
    annotated = image.copy()
    
    # Iterate through each detection and draw its bounding box
    for det in detections:
        # Extract bounding box coordinates from detection dict
        bbox = det.get("bbox", {})
        x1 = int(bbox.get("x1", 0))  # Top-left x coordinate
        y1 = int(bbox.get("y1", 0))  # Top-left y coordinate
        x2 = int(bbox.get("x2", 0))  # Bottom-right x coordinate
        y2 = int(bbox.get("y2", 0))  # Bottom-right y coordinate
        
        # Determine damage type and select appropriate color for visualization
        damage_type = det.get("damage_type") or det.get("class", "pothole")
        color = DAMAGE_COLORS.get(damage_type, DEFAULT_BOX_COLOR)  # BGR format
        
        # Draw the main bounding box rectangle around the detection
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Build label text (damage type and optionally confidence)
        confidence = det.get("confidence", 0)
        if show_confidence:
            label = f"{damage_type}: {confidence:.0%}"  # Format confidence as percentage
        else:
            label = damage_type
        
        # Calculate dimensions of label text to create background box
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Position label above the bounding box (with 10 pixel margin)
        label_y1 = max(y1 - label_h - 10, 0)  # Ensure label stays within image bounds
        label_y2 = y1
        
        # Draw colored background rectangle behind label text
        cv2.rectangle(annotated, (x1, label_y1), (x1 + label_w + 4, label_y2), color, -1)  # -1 fills the rectangle
        
        # Draw label text in white color on colored background
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y2 - 4),  # Position text with small padding
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White color in BGR format
            thickness - 1 if thickness > 1 else 1,  # Slightly thinner text for readability
            cv2.LINE_AA,  # Use antialiasing for smoother text
        )
    
    return annotated


def save_annotated_image(
    image_path: str,
    detections: List[Dict],
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Save an image with bounding boxes drawn
    
    This function reads an image file, draws bounding boxes for all detections on it,
    and saves the annotated result to disk. It provides flexible output path handling:
    explicit output_path, output_dir with auto-naming, or same directory as input.
    
    Args:
        image_path: Path to original image file to annotate
        detections: List of detection dictionaries with bbox and damage_type info
        output_path: Explicit output file path (takes precedence over output_dir)
        output_dir: Directory to save output (uses original filename with _annotated suffix)
    
    Returns:
        Path to the saved annotated image file
    
    Raises:
        ValueError: If the image file cannot be read
    """
    # Load the image from disk using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Draw bounding boxes and labels on the image
    annotated = draw_bounding_boxes(image, detections)
    
    # Determine output file path based on provided arguments
    if output_path is None:
        # Parse input file path to extract directory, filename, and extension
        input_path = Path(image_path)
        
        if output_dir:
            # Create output directory if it doesn't exist
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Build output filename: original_name_annotated.extension
            output_path = str(out_dir / f"{input_path.stem}_annotated{input_path.suffix}")
        else:
            # Save in same directory as input image with _annotated suffix
            output_path = str(input_path.parent / f"{input_path.stem}_annotated{input_path.suffix}")
    
    # Write annotated image to disk
    cv2.imwrite(output_path, annotated)
    logger.info(f"Saved annotated image: {output_path}")
    
    return output_path


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
    
    This is the main inference function that processes a batch of image frames through
    the YOLO model to detect potholes and other road damage. It handles batch processing
    for efficiency, loads the model on first call, and returns structured detection results.
    
    Args:
        frame_paths: List of paths to frame images to process
        conf_threshold: Minimum confidence threshold for detections (0.0-1.0)
        timestamps: Optional list of timestamps (in seconds) for each frame
        model_path: Optional path to model (local .pt file or HuggingFace model ID)
        iou_threshold: IoU (Intersection over Union) threshold for NMS (Non-Maximum Suppression)
    
    Returns:
        List of RawDetection objects containing detection information
    """
    # Validate input - return empty list if no frames provided
    if not frame_paths:
        logger.warning("No frame paths provided for inference")
        return []
    
    # Generate default timestamps (0, 1, 2, ...) if not provided
    if timestamps is None:
        timestamps = [float(i) for i in range(len(frame_paths))]
    
    # Validate and correct timestamp list to match number of frames
    if len(timestamps) != len(frame_paths):
        logger.warning(f"Timestamp count ({len(timestamps)}) != frame count ({len(frame_paths)})")
        timestamps = [float(i) for i in range(len(frame_paths))]
    
    # Load the YOLO model (cached after first load)
    model = _load_model(model_path)
    
    # Initialize results container
    detections: List[RawDetection] = []
    
    # Log inference configuration
    logger.info(f"Running inference on {len(frame_paths)} frames...")
    logger.info(f"  Confidence threshold: {conf_threshold}")
    logger.info(f"  IoU threshold: {iou_threshold}")
    
    # Set batch size for processing efficiency (process 16 frames at a time)
    batch_size = 16
    total_detections = 0
    
    # Process frames in batches to maximize GPU/CPU efficiency
    for batch_start in range(0, len(frame_paths), batch_size):
        # Calculate batch boundaries
        batch_end = min(batch_start + batch_size, len(frame_paths))
        batch_paths = frame_paths[batch_start:batch_end]
        batch_timestamps = timestamps[batch_start:batch_end]
        
        try:
            # Run YOLO inference on the batch of frames
            # Returns a list of Result objects, one per frame
            results = model(
                batch_paths,
                conf=conf_threshold,  # Filter detections below this confidence
                iou=iou_threshold,    # NMS threshold to remove duplicate detections
                verbose=False,        # Suppress YOLO's verbose output
            )
            
            # Process the inference results for each frame
            for idx, result in enumerate(results):
                frame_path = batch_paths[idx]
                timestamp = batch_timestamps[idx]
                
                # Extract original frame dimensions (before any resizing by YOLO)
                frame_height, frame_width = result.orig_shape
                
                # Process detected bounding boxes if any exist
                if result.boxes is not None and len(result.boxes) > 0:
                    # Iterate through each detected box in the frame
                    for box in result.boxes:
                        # Extract bounding box coordinates in xyxy format (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Extract confidence score (probability) for this detection
                        confidence = box.conf[0].item()
                        
                        # Extract class ID and map it to class name
                        class_id = int(box.cls[0].item())
                        class_names = result.names  # Dictionary mapping class IDs to names
                        original_class = class_names.get(class_id, "pothole")  # Fallback to pothole if unknown
                        
                        # Normalize the class name to a standard damage type (pothole, crack, etc.)
                        damage_type = _normalize_class(original_class)
                        
                        # Create a RawDetection object with all relevant information
                        detection = RawDetection(
                            frame_path=frame_path,
                            timestamp=timestamp,
                            damage_type=damage_type,  # Standardized damage type
                            confidence=round(confidence, 4),  # Round to 4 decimal places
                            bbox={  # Bounding box coordinates
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                            },
                            frame_width=frame_width,
                            frame_height=frame_height,
                            original_class=original_class,  # Keep original class for reference
                        )
                        
                        # Add detection to results list
                        detections.append(detection)
                        total_detections += 1
            
            # Log progress periodically (every 50 frames or at the end)
            if (batch_end) % 50 == 0 or batch_end == len(frame_paths):
                logger.info(f"  Processed {batch_end}/{len(frame_paths)} frames, {total_detections} detections so far")
                
        except Exception as e:
            # Log errors but continue processing remaining batches
            logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
            continue
    
    # Log completion summary
    logger.info(f"Inference complete: {len(detections)} detections from {len(frame_paths)} frames")
    
    return detections


def detect_single_image(
    image_path: str,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_path: Optional[str] = None,
    save_annotated: bool = False,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str]]:
    """Convenience function to detect potholes in a single image
    
    This is a high-level wrapper around run_inference for processing a single image.
    It automatically converts RawDetection objects to dictionaries, calculates bounding
    box areas, and optionally saves an annotated visualization of the detections.
    
    Args:
        image_path: Path to image file to process
        conf_threshold: Minimum confidence threshold (0.0-1.0) for detections
        model_path: Optional path to model (local .pt file or HuggingFace model ID)
        save_annotated: Whether to save an annotated image with bounding boxes drawn
        output_path: Explicit file path for annotated output image
        output_dir: Directory where annotated image should be saved (uses original filename)
    
    Returns:
        Tuple of:
            - List of detection dictionaries with bbox, confidence, damage_type, etc.
            - Path to annotated image file if save_annotated is True, otherwise None
    """
    # Run inference on the single image
    detections = run_inference(
        frame_paths=[image_path],
        conf_threshold=conf_threshold,
        timestamps=[0.0],  # Single frame, timestamp is irrelevant but required
        model_path=model_path,
    )
    
    # Convert RawDetection objects to dictionary format for easier use
    results = []
    for det in detections:
        bbox = det.bbox
        # Calculate detection area in pixels (width * height)
        area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
        
        # Create result dictionary with all relevant detection information
        results.append({
            "class": det.damage_type,  # Normalized damage class
            "damage_type": det.damage_type,  # Normalized damage class (duplicate for compatibility)
            "original_class": det.original_class,  # Original class from model
            "confidence": det.confidence,  # Detection confidence score
            "bbox": bbox,  # Bounding box coordinates {x1, y1, x2, y2}
            "area": area,  # Pixel area of detection
            "frame_width": det.frame_width,  # Original image width
            "frame_height": det.frame_height,  # Original image height
        })
    
    # Optionally save an annotated image with bounding boxes drawn
    annotated_path = None
    if save_annotated and results:
        annotated_path = save_annotated_image(
            image_path,
            results,
            output_path=output_path,
            output_dir=output_dir,
        )
    
    return results, annotated_path


if __name__ == "__main__":
    # Test inference on a single image
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pothole detection")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", "-s", action="store_true", help="Save annotated image with bounding boxes")
    parser.add_argument("--output", "-o", help="Output path for annotated image")
    parser.add_argument("--output-dir", "-d", help="Output directory for annotated image")
    args = parser.parse_args()
    
    print(f"Testing inference on: {args.image}")
    
    results, annotated_path = detect_single_image(
        args.image,
        conf_threshold=args.conf,
        save_annotated=args.save or args.output is not None or args.output_dir is not None,
        output_path=args.output,
        output_dir=args.output_dir,
    )
    
    if results:
        print(f"\nFound {len(results)} pothole(s):")
        for i, det in enumerate(results, 1):
            print(f"  {i}. {det['class']} (conf: {det['confidence']:.2%})")
            print(f"     Bbox: {det['bbox']}")
            print(f"     Area: {det['area']} pixels")
        
        if annotated_path:
            print(f"\nAnnotated image saved to: {annotated_path}")
    else:
        print("\nNo potholes detected.")
