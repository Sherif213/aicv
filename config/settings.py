import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

for directory in [DATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

class HardwareConfig:
    CAMERA_INDEX = 0
    
    ARDUINO_PORT = "/dev/ttyUSB0"
    ARDUINO_BAUDRATE = 115200
    
    MOTOR_LEFT_FORWARD = 8
    MOTOR_LEFT_BACKWARD = 9
    MOTOR_RIGHT_FORWARD = 10
    MOTOR_RIGHT_BACKWARD = 11
    SERVO_STEERING_PIN = 12
    
    ULTRASONIC_FRONT_TRIG = 3
    ULTRASONIC_FRONT_ECHO = 2
    ULTRASONIC_LEFT_TRIG = 7
    ULTRASONIC_LEFT_ECHO = 5
    ULTRASONIC_RIGHT_TRIG = 6
    ULTRASONIC_RIGHT_ECHO = 4
    
    LIDAR_PORT = "/dev/ttyUSB1"
    LIDAR_BAUDRATE = 115200

class ModelConfig:
    YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD = 0.5
    YOLO_NMS_THRESHOLD = 0.4
    
    DETECTION_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
        'traffic light', 'stop sign', 'parking meter', 'bench'
    ]
    
    GRID_SIZE = 50
    MAX_GRID_SIZE = 100
    
    LSTM_SEQUENCE_LENGTH = 10
    PREDICTION_HORIZON = 5

class ControlConfig:
    VEHICLE_LENGTH = 30
    VEHICLE_WIDTH = 20
    WHEELBASE = 25
    
    MAX_SPEED = 100
    MIN_SPEED = 10
    TURNING_SPEED = 50
    
    SAFE_DISTANCE = 100
    WARNING_DISTANCE = 150
    EMERGENCY_STOP_DISTANCE = 30
    
    STEERING_KP = 0.8
    STEERING_KI = 0.1
    STEERING_KD = 0.2
    
    SPEED_KP = 0.6
    SPEED_KI = 0.05
    SPEED_KD = 0.1

class SensorConfig:
    ULTRASONIC_TIMEOUT = 25000
    ULTRASONIC_MAX_DISTANCE = 400
    ULTRASONIC_MIN_DISTANCE = 2
    ULTRASONIC_SAMPLE_RATE = 10
    
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    LIDAR_SCAN_RATE = 10
    LIDAR_MAX_RANGE = 12000
    LIDAR_MIN_RANGE = 20

class CommunicationConfig:
    SERIAL_TIMEOUT = 1.0
    SERIAL_BUFFER_SIZE = 1024
    
    ULTRASONIC_MSG_FORMAT = "L:{left} F:{front} R:{right}"
    MOTOR_CMD_FORMAT = "MOTOR:L:{left_speed}:R:{right_speed}"
    SERVO_CMD_FORMAT = "SERVO:{angle}"
    
    COMMAND_TIMEOUT = 0.5
    HEARTBEAT_INTERVAL = 1.0

class LoggingConfig:
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "autonomous_vehicle.log"
    
    PERFORMANCE_LOG_FILE = LOGS_DIR / "performance.log"
    DETECTION_LOG_FILE = LOGS_DIR / "detection.log"
    SENSOR_LOG_FILE = LOGS_DIR / "sensor.log"

class SimulationConfig:
    CARLA_HOST = "localhost"
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0
    
    SIM_FPS = 30
    SIM_WEATHER = "ClearNoon"
    SIM_VEHICLE_MODEL = "vehicle.tesla.model3"
    
    TEST_SCENARIOS = [
        "straight_road",
        "intersection",
        "parking_lot",
        "obstacle_avoidance"
    ] 