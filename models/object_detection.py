"""
Object Detection Module using YOLO
Real-time object detection for autonomous vehicle
"""

import cv2
import numpy as np
import torch
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import threading

from config.settings import ModelConfig, SensorConfig
from core.logger import logger


@dataclass
class Detection:
    """Data structure for object detection results"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]  # center point
    distance: Optional[float] = None  # estimated distance


class ObjectDetector:
    """YOLO-based object detector for autonomous vehicle"""
    
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or str(ModelConfig.YOLO_MODEL_PATH)
        self.device = device or self._select_device()
        
        # Model and inference
        self.model: Optional[torch.nn.Module] = None
        self.class_names: List[str] = []
        self.is_loaded = False
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.detection_count = 0
        
        # Threading for real-time processing
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.latest_detections: List[Detection] = []
        self.lock = threading.Lock()
        
        # Camera calibration for distance estimation
        self.focal_length = 615  # pixels (needs calibration)
        self.known_width = 50  # cm (average car width)
        
        logger.info(f"Object detector initialized with device: {self.device}")
    
    def _select_device(self) -> str:
        """Select the best available device for inference"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> bool:
        """Load YOLO model using Ultralytics YOLOv8 API"""
        try:
            logger.start_timer("model_loading")
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            self.is_loaded = True
            loading_time = logger.end_timer("model_loading")
            logger.info(f"YOLO model loaded successfully in {loading_time:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO inference"""
        # Resize image to YOLO input size
        input_size = (640, 640)
        resized = cv2.resize(image, input_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        return rgb_image
    
    def detect_objects(self, image: np.ndarray) -> List[Detection]:
        """Perform object detection on image"""
        if not self.is_loaded:
            logger.error("Model not loaded")
            return []
        try:
            logger.start_timer("inference")
            results = self.model(image)
            inference_time = logger.end_timer("inference")
            self.inference_times.append(inference_time)
            detections = self._parse_detections(results, image.shape)
            logger.log_detection(detections, inference_time)
            self.detection_count += len(detections)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            return detections
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
    
    def _parse_detections(self, results, original_shape: Tuple[int, int, int]) -> List[Detection]:
        """Parse YOLOv8 detection results"""
        detections = []
        try:
            # YOLOv8: results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls
            if hasattr(results, "boxes") or (isinstance(results, list) and hasattr(results[0], "boxes")):
                # Handle both single and batch
                res = results[0] if isinstance(results, list) else results
                boxes = res.boxes
                if boxes is not None and boxes.xyxy is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy()
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        confidence = float(confs[i])
                        class_id = int(clss[i])
                        if confidence < ModelConfig.YOLO_CONFIDENCE_THRESHOLD:
                            continue
                        if class_id >= len(self.class_names):
                            continue
                        class_name = self.class_names[class_id]
                        if class_name not in ModelConfig.DETECTION_CLASSES:
                            continue
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        distance = self._estimate_distance(bbox, class_name)
                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            center=center,
                            distance=distance
                        )
                        detections.append(detection)
        except Exception as e:
            logger.error(f"Error parsing detections: {e}")
        return detections
    
    def _estimate_distance(self, bbox: Tuple[int, int, int, int], class_name: str) -> Optional[float]:
        """Estimate distance to detected object using bbox size"""
        try:
            x, y, width, height = bbox
            
            # Use width for distance estimation
            if width <= 0:
                return None
            
            # Known object widths (in cm) for different classes
            known_widths = {
                'person': 50,
                'car': 180,
                'truck': 250,
                'bus': 250,
                'motorcycle': 80,
                'bicycle': 60,
                'traffic light': 30,
                'stop sign': 60,
                'parking meter': 20,
                'bench': 120
            }
            
            known_width = known_widths.get(class_name, self.known_width)
            
            # Distance = (known_width * focal_length) / pixel_width
            distance = (known_width * self.focal_length) / width
            
            # Validate distance (reasonable range: 1-50 meters)
            if 100 <= distance <= 5000:
                return distance
            
            return None
            
        except Exception as e:
            logger.error(f"Error estimating distance: {e}")
            return None
    
    def start_real_time_detection(self, camera_source: int = 0):
        """Start real-time object detection in background thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Detection thread already running")
            return False
        
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._real_time_detection_loop,
            args=(camera_source,),
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Started real-time object detection")
        return True
    
    def stop_real_time_detection(self):
        """Stop real-time object detection"""
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        logger.info("Stopped real-time object detection")
    
    def _real_time_detection_loop(self, camera_source: int):
        """Background thread for real-time detection"""
        cap = cv2.VideoCapture(camera_source)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SensorConfig.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SensorConfig.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, SensorConfig.CAMERA_FPS)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        logger.info("Camera opened successfully")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                # Perform detection
                detections = self.detect_objects(frame)
                
                # Update latest detections
                with self.lock:
                    self.latest_detections = detections
                
                # Add small delay to prevent overwhelming the system
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error in real-time detection loop: {e}")
        
        finally:
            cap.release()
            logger.info("Camera released")
    
    def get_latest_detections(self) -> List[Detection]:
        """Get latest detection results"""
        with self.lock:
            return self.latest_detections.copy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        if not self.inference_times:
            return {}
        
        avg_inference_time = np.mean(self.inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            "avg_inference_time": avg_inference_time,
            "fps": fps,
            "total_detections": self.detection_count,
            "detection_rate": self.detection_count / max(len(self.inference_times), 1)
        }
    
    def calibrate_distance(self, known_distance: float, known_width: float, bbox_width: int):
        """Calibrate distance estimation"""
        if bbox_width > 0:
            self.focal_length = (bbox_width * known_distance) / known_width
            logger.info(f"Distance calibration updated: focal_length = {self.focal_length}")
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection results on image"""
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            color = (0, 255, 0) if detection.confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if detection.distance:
                label += f" ({detection.distance:.1f}cm)"
            
            # Calculate text position
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x
            text_y = y - 10 if y - 10 > text_size[1] else y + h + text_size[1]
            
            # Draw text background
            cv2.rectangle(image, (text_x, text_y - text_size[1]), 
                         (text_x + text_size[0], text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(image, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image 