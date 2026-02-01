"""YOLO inference module - runs object detection on frames"""

import sys
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import cv2

# add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from config import YOLO_MODEL_PATH, CLASS_MAPPING, DEFAULT_CONFIDENCE_THRESHOLD
except ImportError:
    YOLO_MODEL_PATH = "./models/weights/best.pt"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    CLASS_MAPPING = {
        "pothole": "pothole",
        "Pothole": "pothole",
        "D00": "crack",
        "D10": "crack",
        "D20": "crack",
        "D40": "pothole",
        "crack": "crack",
        "rut": "rut",
        "debris": "debris",
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

from ultralytics import YOLO


@dataclass
class RawDetection:
    """raw detection result from yolo inference"""
    frame_path: str
    timestamp: float
    damage_type: str
    confidence: float
    bbox: Dict[str, int]
    frame_width: int
    frame_height: int
    original_class: str = ""

    def to_dict(self) -> Dict:
        """convert to dict for json"""
        return {
            "frame_path": self.frame_path,
            "timestamp": self.timestamp,
            "damage_type": self.damage_type,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "original_class": self.original_class,
        }


class InferenceEngine:
    """yolo inference engine for road damage detection"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        self.model_path = model_path or YOLO_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.model: Optional[YOLO] = None
        self.class_names: Dict = {}
        
    def load_model(self) -> None:
        """load yolo model from disk"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        
        logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        #get class names from model
        self.class_names = self.model.names
        logger.info(f"Model classes: {self.class_names}")
    
    def _map_class(self, class_name: str) -> str:
        """map yolo class name to our damage types"""
        mapped = CLASS_MAPPING.get(class_name, class_name.lower())
        
        #ensure valid damage type
        valid_types = {"pothole", "crack", "rut", "debris"}
        if mapped not in valid_types:
            logger.warning(f"Unknown class '{class_name}' mapped to 'pothole'")
            return "pothole"
        
        return mapped
    
    def run_inference(
        self,
        frame_paths: List[str],
        timestamps: Optional[List[float]] = None
    ) -> List[RawDetection]:
        """run inference on list of frames"""
        if self.model is None:
            self.load_model()
        
        if timestamps is None:
            timestamps = [0.0] * len(frame_paths)
        
        if len(timestamps) != len(frame_paths):
            raise ValueError("timestamps length must match frame_paths length")
        
        all_detections: List[RawDetection] = []
        
        logger.info(f"Running inference on {len(frame_paths)} frames...")
        
        for idx, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps)):
            #read frame to get dimensions
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Cannot read frame: {frame_path}")
                continue
            
            height, width = frame.shape[:2]
            
            #run yolo inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            #process detections
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                for i in range(len(boxes)):
                    #get bounding box coords (xyxy format)
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    #get confidence and class
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    #get class name
                    original_class = self.class_names.get(cls_id, str(cls_id))
                    damage_type = self._map_class(original_class)
                    
                    detection = RawDetection(
                        frame_path=frame_path,
                        timestamp=timestamp,
                        damage_type=damage_type,
                        confidence=round(conf, 4),
                        bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        frame_width=width,
                        frame_height=height,
                        original_class=original_class,
                    )
                    
                    all_detections.append(detection)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(frame_paths)} frames, {len(all_detections)} detections")
        
        logger.info(f"Inference complete: {len(all_detections)} total detections")
        
        return all_detections


def run_inference(
    frame_paths: List[str],
    model_path: Optional[str] = None,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    timestamps: Optional[List[float]] = None
) -> List[RawDetection]:
    """convenience function to run inference"""
    engine = InferenceEngine(model_path=model_path, conf_threshold=conf_threshold)
    return engine.run_inference(frame_paths, timestamps)
