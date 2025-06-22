"""
Configuration settings for AI-Based Autonomous Vehicle System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Hardware Configuration
class HardwareConfig:
    # Raspberry Pi GPIO pins
    CAMERA_INDEX = 0  # USB camera index
    
    # Arduino communication
    ARDUINO_PORT = "/dev/ttyUSB0"  # USB1 Arduino
    ARDUINO_BAUDRATE = 115200
    
    # Motor control pins (via Arduino)
    MOTOR_LEFT_FORWARD = 8
    MOTOR_LEFT_BACKWARD = 9
    MOTOR_RIGHT_FORWARD = 10
    MOTOR_RIGHT_BACKWARD = 11
    SERVO_STEERING_PIN = 12
    
    # Sensor pins (via Arduino)
    ULTRASONIC_FRONT_TRIG = 3
    ULTRASONIC_FRONT_ECHO = 2
    ULTRASONIC_LEFT_TRIG = 7
    ULTRASONIC_LEFT_ECHO = 5
    ULTRASONIC_RIGHT_TRIG = 6
    ULTRASONIC_RIGHT_ECHO = 4
    
    # LiDAR configuration (if using)
    LIDAR_PORT = "/dev/ttyUSB1"  # USB2 LiDAR
    LIDAR_BAUDRATE = 115200

# AI Model Configuration
class ModelConfig:
    # YOLO configuration
    YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD = 0.5
    YOLO_NMS_THRESHOLD = 0.4
    
    # Object detection classes of interest
    DETECTION_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
        'traffic light', 'stop sign', 'parking meter', 'bench'
    ]
    
    # Path planning
    GRID_SIZE = 50  # cm per grid cell
    MAX_GRID_SIZE = 100  # maximum grid size for A* search
    
    # Behavior prediction
    LSTM_SEQUENCE_LENGTH = 10
    PREDICTION_HORIZON = 5

# Control System Configuration
class ControlConfig:
    # Vehicle parameters
    VEHICLE_LENGTH = 30  # cm
    VEHICLE_WIDTH = 20   # cm
    WHEELBASE = 25       # cm
    
    # Speed limits
    MAX_SPEED = 100      # cm/s
    MIN_SPEED = 10       # cm/s
    TURNING_SPEED = 50   # cm/s
    
    # Safety thresholds
    SAFE_DISTANCE = 100  # cm
    WARNING_DISTANCE = 150  # cm
    EMERGENCY_STOP_DISTANCE = 30  # cm
    
    # Control gains
    STEERING_KP = 0.8
    STEERING_KI = 0.1
    STEERING_KD = 0.2
    
    SPEED_KP = 0.6
    SPEED_KI = 0.05
    SPEED_KD = 0.1

# Sensor Configuration
class SensorConfig:
    # Ultrasonic sensors
    ULTRASONIC_TIMEOUT = 25000  # microseconds
    ULTRASONIC_MAX_DISTANCE = 400  # cm
    ULTRASONIC_MIN_DISTANCE = 2    # cm
    ULTRASONIC_SAMPLE_RATE = 10    # Hz
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # LiDAR settings
    LIDAR_SCAN_RATE = 10  # Hz
    LIDAR_MAX_RANGE = 12000  # mm
    LIDAR_MIN_RANGE = 20    # mm

# Communication Configuration
class CommunicationConfig:
    # Serial communication
    SERIAL_TIMEOUT = 1.0  # seconds
    SERIAL_BUFFER_SIZE = 1024
    
    # Message formats
    ULTRASONIC_MSG_FORMAT = "L:{left} F:{front} R:{right}"
    MOTOR_CMD_FORMAT = "MOTOR:L:{left_speed}:R:{right_speed}"
    SERVO_CMD_FORMAT = "SERVO:{angle}"
    
    # Command timeouts
    COMMAND_TIMEOUT = 0.5  # seconds
    HEARTBEAT_INTERVAL = 1.0  # seconds

# Logging Configuration
class LoggingConfig:
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "autonomous_vehicle.log"
    
    # Performance logging
    PERFORMANCE_LOG_FILE = LOGS_DIR / "performance.log"
    DETECTION_LOG_FILE = LOGS_DIR / "detection.log"
    SENSOR_LOG_FILE = LOGS_DIR / "sensor.log"

# Simulation Configuration
class SimulationConfig:
    # CARLA simulation settings
    CARLA_HOST = "localhost"
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0
    
    # Simulation parameters
    SIM_FPS = 30
    SIM_WEATHER = "ClearNoon"
    SIM_VEHICLE_MODEL = "vehicle.tesla.model3"
    
    # Test scenarios
    TEST_SCENARIOS = [
        "straight_road",
        "intersection",
        "parking_lot",
        "obstacle_avoidance"
    ] 