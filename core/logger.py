"""
Logging system for AI-Based Autonomous Vehicle
"""

import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

from config.settings import LoggingConfig


class VehicleLogger:
    """Centralized logging system for the autonomous vehicle"""
    
    def __init__(self, name: str = "autonomous_vehicle"):
        self.name = name
        self.logger = self._setup_logger()
        self.performance_logger = self._setup_performance_logger()
        self.detection_logger = self._setup_detection_logger()
        self.sensor_logger = self._setup_sensor_logger()
        self.behavior_logger = self._setup_behavior_logger()
        
        # Performance tracking
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup main application logger"""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, LoggingConfig.LOG_LEVEL))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(LoggingConfig.LOG_FILE)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(LoggingConfig.LOG_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance metrics logger"""
        logger = logging.getLogger(f"{self.name}.performance")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        file_handler = logging.FileHandler(LoggingConfig.PERFORMANCE_LOG_FILE)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_detection_logger(self) -> logging.Logger:
        """Setup object detection logger"""
        logger = logging.getLogger(f"{self.name}.detection")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        file_handler = logging.FileHandler(LoggingConfig.DETECTION_LOG_FILE)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_sensor_logger(self) -> logging.Logger:
        """Setup sensor data logger"""
        logger = logging.getLogger(f"{self.name}.sensor")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        file_handler = logging.FileHandler(LoggingConfig.SENSOR_LOG_FILE)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_behavior_logger(self) -> logging.Logger:
        """Setup behavior prediction logger"""
        logger = logging.getLogger(f"{self.name}.behavior")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        behavior_log_file = Path("logs/behavior_prediction.log")
        behavior_log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(behavior_log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        self.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation not in self.start_times:
            self.warning(f"Timer for {operation} was not started")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.performance_logger.info(f"{operation}: {duration:.4f}s")
        del self.start_times[operation]
        return duration
    
    def log_detection(self, detections: list, frame_time: float):
        """Log object detection results"""
        detection_data = {
            "timestamp": datetime.now().isoformat(),
            "frame_time": frame_time,
            "detections": detections
        }
        self.detection_logger.info(json.dumps(detection_data))
    
    def log_sensor_data(self, sensor_type: str, data: Dict[str, Any]):
        """Log sensor data"""
        sensor_data = {
            "timestamp": datetime.now().isoformat(),
            "sensor_type": sensor_type,
            "data": data
        }
        self.sensor_logger.info(json.dumps(sensor_data))
    
    def log_behavior_prediction(self, prediction, prediction_time: float):
        """Log behavior prediction results"""
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "object_id": prediction.object_id,
            "class_name": prediction.class_name,
            "current_position": prediction.current_position,
            "predicted_speed": prediction.predicted_speed,
            "predicted_direction": prediction.predicted_direction,
            "confidence": prediction.confidence,
            "prediction_horizon": prediction.prediction_horizon,
            "prediction_time": prediction_time,
            "trajectory_points": len(prediction.predicted_trajectory)
        }
        self.behavior_logger.info(json.dumps(prediction_data))
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metric"""
        self.performance_logger.info(f"{metric_name}: {value}{unit}")
        self.metrics[metric_name] = value
    
    def log_system_status(self, status: Dict[str, Any]):
        """Log system status"""
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "status": status
        }
        self.logger.info(f"System Status: {json.dumps(status_data)}")
    
    def log_emergency(self, emergency_type: str, details: str):
        """Log emergency situations"""
        emergency_data = {
            "timestamp": datetime.now().isoformat(),
            "emergency_type": emergency_type,
            "details": details
        }
        self.error(f"EMERGENCY: {json.dumps(emergency_data)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def export_logs(self, output_path: Path):
        """Export all logs to a single file"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = output_path / f"vehicle_logs_{timestamp}.txt"
        
        with open(export_file, 'w') as f:
            # Main log
            if LoggingConfig.LOG_FILE.exists():
                f.write("=== MAIN LOG ===\n")
                f.write(LoggingConfig.LOG_FILE.read_text())
                f.write("\n\n")
            
            # Performance log
            if LoggingConfig.PERFORMANCE_LOG_FILE.exists():
                f.write("=== PERFORMANCE LOG ===\n")
                f.write(LoggingConfig.PERFORMANCE_LOG_FILE.read_text())
                f.write("\n\n")
            
            # Detection log
            if LoggingConfig.DETECTION_LOG_FILE.exists():
                f.write("=== DETECTION LOG ===\n")
                f.write(LoggingConfig.DETECTION_LOG_FILE.read_text())
                f.write("\n\n")
            
            # Sensor log
            if LoggingConfig.SENSOR_LOG_FILE.exists():
                f.write("=== SENSOR LOG ===\n")
                f.write(LoggingConfig.SENSOR_LOG_FILE.read_text())
        
        self.info(f"Logs exported to: {export_file}")
        return export_file


# Global logger instance
logger = VehicleLogger() 