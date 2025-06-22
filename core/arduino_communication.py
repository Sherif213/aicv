import serial
import time
import threading
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import re

from config.settings import HardwareConfig, CommunicationConfig, SensorConfig
from core.logger import logger


@dataclass
class UltrasonicData:
    front: float
    left: float
    right: float
    timestamp: float


@dataclass
class MotorCommand:
    left_speed: int
    right_speed: int
    steering_angle: int


class ArduinoCommunication:
    
    def __init__(self, port: Optional[str] = None, baudrate: Optional[int] = None):
        self.port = port or HardwareConfig.ARDUINO_PORT
        self.baudrate = baudrate or HardwareConfig.ARDUINO_BAUDRATE
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_running = False
        
        self.latest_ultrasonic_data: Optional[UltrasonicData] = None
        self.latest_motor_status: Optional[MotorCommand] = None
        
        self.read_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.last_heartbeat = 0
        self.command_queue: List[str] = []
        
        logger.info(f"Arduino communication initialized for port: {self.port}")
    
    def connect(self) -> bool:
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=CommunicationConfig.SERIAL_TIMEOUT,
                write_timeout=CommunicationConfig.SERIAL_TIMEOUT
            )
            
            time.sleep(2)
            
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            
            startup_timeout = 10
            start_time = time.time()
            
            while time.time() - start_time < startup_timeout:
                if self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if "RPI_ULTRASONIC_STARTED" in line:
                        self.is_connected = True
                        logger.info("Arduino connection established successfully")
                        return True
                time.sleep(0.1)
            
            logger.error("Arduino startup message not received")
            return False
            
        except serial.SerialException as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Arduino connection: {e}")
            return False
    
    def disconnect(self):
        self.is_running = False
        self.is_connected = False
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2)
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        
        logger.info("Arduino connection closed")
    
    def start_reading(self):
        if not self.is_connected:
            logger.error("Cannot start reading: Arduino not connected")
            return False
        
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        logger.info("Started Arduino reading thread")
        return True
    
    def _read_loop(self):
        while self.is_running and self.is_connected:
            try:
                if self.serial_connection and self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if line:
                        self._parse_arduino_message(line)
                
                self._send_queued_commands()
                
                self._send_heartbeat()
                
                time.sleep(0.01)
                
            except serial.SerialException as e:
                logger.error(f"Serial communication error: {e}")
                self.is_connected = False
                break
            except Exception as e:
                logger.error(f"Error in Arduino read loop: {e}")
                time.sleep(0.1)
    
    def _parse_arduino_message(self, message: str):
        try:
            if message.startswith("L:") and "F:" in message and "R:" in message:
                self._parse_ultrasonic_data(message)
            
            elif message.startswith("MOTOR_STATUS:"):
                self._parse_motor_status(message)
            
            elif message.startswith("ERROR:"):
                logger.error(f"Arduino error: {message}")
            
            elif message == "HEARTBEAT":
                self.last_heartbeat = time.time()
            
        except Exception as e:
            logger.error(f"Error parsing Arduino message '{message}': {e}")
    
    def _parse_ultrasonic_data(self, message: str):
        try:
            pattern = r"L:([\d.-]+)\s+F:([\d.-]+)\s+R:([\d.-]+)"
            match = re.match(pattern, message)
            
            if match:
                left = float(match.group(1))
                front = float(match.group(2))
                right = float(match.group(3))
                
                if (SensorConfig.ULTRASONIC_MIN_DISTANCE <= left <= SensorConfig.ULTRASONIC_MAX_DISTANCE and
                    SensorConfig.ULTRASONIC_MIN_DISTANCE <= front <= SensorConfig.ULTRASONIC_MAX_DISTANCE and
                    SensorConfig.ULTRASONIC_MIN_DISTANCE <= right <= SensorConfig.ULTRASONIC_MAX_DISTANCE):
                    
                    with self.lock:
                        self.latest_ultrasonic_data = UltrasonicData(
                            front=front,
                            left=left,
                            right=right,
                            timestamp=time.time()
                        )
                    
                    logger.log_sensor_data("ultrasonic", {
                        "front": front,
                        "left": left,
                        "right": right
                    })
                else:
                    logger.warning(f"Invalid ultrasonic distances: {message}")
            
        except ValueError as e:
            logger.error(f"Invalid ultrasonic data format: {message}, error: {e}")
    
    def _parse_motor_status(self, message: str):
        try:
            pattern = r"MOTOR_STATUS:L:([\d-]+):R:([\d-]+):S:([\d-]+)"
            match = re.match(pattern, message)
            
            if match:
                left_speed = int(match.group(1))
                right_speed = int(match.group(2))
                steering_angle = int(match.group(3))
                
                with self.lock:
                    self.latest_motor_status = MotorCommand(
                        left_speed=left_speed,
                        right_speed=right_speed,
                        steering_angle=steering_angle
                    )
        
        except ValueError as e:
            logger.error(f"Invalid motor status format: {message}, error: {e}")
    
    def get_ultrasonic_data(self) -> Optional[UltrasonicData]:
        with self.lock:
            return self.latest_ultrasonic_data
    
    def get_motor_status(self) -> Optional[MotorCommand]:
        with self.lock:
            return self.latest_motor_status
    
    def send_motor_command(self, left_speed: int, right_speed: int, steering_angle: int = 90):
        if not self.is_connected:
            logger.error("Cannot send motor command: Arduino not connected")
            return False
        
        left_speed = max(-255, min(255, left_speed))
        right_speed = max(-255, min(255, right_speed))
        steering_angle = max(0, min(180, steering_angle))
        
        command = CommunicationConfig.MOTOR_CMD_FORMAT.format(
            left_speed=left_speed,
            right_speed=right_speed
        )
        
        steering_command = CommunicationConfig.SERVO_CMD_FORMAT.format(angle=steering_angle)
        
        self.command_queue.append(command)
        self.command_queue.append(steering_command)
        
        logger.debug(f"Queued motor command: L={left_speed}, R={right_speed}, S={steering_angle}")
        return True
    
    def _send_queued_commands(self):
        if not self.command_queue or not self.is_connected or not self.serial_connection:
            return
        
        try:
            while self.command_queue:
                command = self.command_queue.pop(0)
                self.serial_connection.write(f"{command}\n".encode('utf-8'))
                self.serial_connection.flush()
                time.sleep(0.01)
        
        except serial.SerialException as e:
            logger.error(f"Error sending command to Arduino: {e}")
            self.is_connected = False
    
    def _send_heartbeat(self):
        if time.time() - self.last_heartbeat > CommunicationConfig.HEARTBEAT_INTERVAL:
            try:
                if self.is_connected and self.serial_connection:
                    self.serial_connection.write("HEARTBEAT\n".encode('utf-8'))
                    self.serial_connection.flush()
                    self.last_heartbeat = time.time()
            except serial.SerialException as e:
                logger.error(f"Error sending heartbeat: {e}")
                self.is_connected = False
    
    def emergency_stop(self):
        if self.is_connected and self.serial_connection:
            try:
                self.serial_connection.write("EMERGENCY_STOP\n".encode('utf-8'))
                self.serial_connection.flush()
                logger.warning("Emergency stop command sent to Arduino")
            except serial.SerialException as e:
                logger.error(f"Error sending emergency stop: {e}")
    
    def is_healthy(self) -> bool:
        if not self.is_connected:
            return False
        
        if time.time() - self.last_heartbeat > CommunicationConfig.HEARTBEAT_INTERVAL * 3:
            logger.warning("Arduino heartbeat timeout")
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected,
            "running": self.is_running,
            "last_heartbeat": self.last_heartbeat,
            "command_queue_size": len(self.command_queue),
            "latest_ultrasonic": self.latest_ultrasonic_data is not None,
            "latest_motor_status": self.latest_motor_status is not None
        } 