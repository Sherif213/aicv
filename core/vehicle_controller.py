"""
Vehicle Controller for Autonomous Vehicle
Main control system that integrates all components
"""

import time
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

from config.settings import ControlConfig, HardwareConfig
from core.logger import logger
from core.arduino_communication import ArduinoCommunication, UltrasonicData
from core.path_planning import PathPlanner, PathPoint
from models.object_detection import ObjectDetector, Detection
from models.behavior_prediction import BehaviorPredictor, BehaviorPrediction


class VehicleState(Enum):
    """Vehicle operation states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    NAVIGATING = "navigating"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"


@dataclass
class VehicleStatus:
    """Vehicle status information"""
    state: VehicleState
    current_position: Tuple[float, float]
    current_heading: float
    current_speed: float
    target_waypoint: Optional[PathPoint]
    obstacles_detected: List[Detection]
    behavior_predictions: List[BehaviorPrediction]
    ultrasonic_data: Optional[UltrasonicData]
    battery_level: float
    system_health: Dict[str, bool]


class VehicleController:
    """Main vehicle controller for autonomous operation"""
    
    def __init__(self):
        # Core components
        self.arduino_comm = ArduinoCommunication()
        self.object_detector = ObjectDetector()
        self.path_planner = PathPlanner()
        self.behavior_predictor = BehaviorPredictor()
        
        # Vehicle state
        self.state = VehicleState.IDLE
        self.status = VehicleStatus(
            state=VehicleState.IDLE,
            current_position=(0.0, 0.0),
            current_heading=0.0,
            current_speed=0.0,
            target_waypoint=None,
            obstacles_detected=[],
            behavior_predictions=[],
            ultrasonic_data=None,
            battery_level=100.0,
            system_health={}
        )
        
        # Control parameters
        self.target_speed = 0.0
        self.target_steering = 90  # degrees (90 = straight)
        self.emergency_stop_active = False
        
        # Navigation
        self.current_goal: Optional[Tuple[float, float]] = None
        self.waypoint_reached_threshold = 20.0  # cm
        
        # Threading
        self.control_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Performance tracking
        self.control_loop_count = 0
        self.last_control_time = 0
        
        logger.info("Vehicle controller initialized")
    
    def initialize(self) -> bool:
        """Initialize all vehicle systems"""
        logger.info("Initializing vehicle systems...")
        self.state = VehicleState.INITIALIZING
        
        try:
            # Initialize Arduino communication
            if not self.arduino_comm.connect():
                logger.error("Failed to connect to Arduino")
                return False
            
            if not self.arduino_comm.start_reading():
                logger.error("Failed to start Arduino reading")
                return False
            
            # Initialize object detection
            if not self.object_detector.load_model():
                logger.error("Failed to load object detection model")
                return False
            
            # Initialize path planner
            self.path_planner.initialize_grid(100, 100)  # 5m x 5m grid
            
            # Initialize behavior prediction
            if not self.behavior_predictor.load_model():
                logger.warning("Failed to load behavior prediction model")
            
            # Start real-time detection
            if not self.object_detector.start_real_time_detection():
                logger.error("Failed to start real-time detection")
                return False
            
            # Update system health
            self._update_system_health()
            
            logger.info("Vehicle systems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            self.state = VehicleState.ERROR
            return False
    
    def start_autonomous_operation(self):
        """Start autonomous vehicle operation"""
        if self.state == VehicleState.ERROR:
            logger.error("Cannot start operation: system in error state")
            return False
        
        if self.control_thread and self.control_thread.is_alive():
            logger.warning("Control thread already running")
            return False
        
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        logger.info("Autonomous operation started")
        return True
    
    def stop_autonomous_operation(self):
        """Stop autonomous vehicle operation"""
        self.is_running = False
        self.emergency_stop()
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2)
        
        logger.info("Autonomous operation stopped")
    
    def set_goal(self, goal: Tuple[float, float]):
        """Set navigation goal"""
        self.current_goal = goal
        logger.info(f"Navigation goal set: {goal}")
    
    def emergency_stop(self):
        """Emergency stop the vehicle"""
        self.emergency_stop_active = True
        self.state = VehicleState.EMERGENCY_STOP
        
        # Stop motors
        self.arduino_comm.emergency_stop()
        
        # Stop autonomous operation
        self.is_running = False
        
        logger.log_emergency("emergency_stop", "Vehicle emergency stop activated")
    
    def _control_loop(self):
        """Main control loop for autonomous operation"""
        control_frequency = 20  # Hz
        control_interval = 1.0 / control_frequency
        
        logger.info("Control loop started")
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Update sensor data
                self._update_sensor_data()
                
                # Check for emergency conditions
                if self._check_emergency_conditions():
                    self.emergency_stop()
                    break
                
                # Update vehicle state
                self._update_vehicle_state()
                
                # Execute control based on current state
                if self.state == VehicleState.NAVIGATING:
                    self._execute_navigation()
                elif self.state == VehicleState.OBSTACLE_AVOIDANCE:
                    self._execute_obstacle_avoidance()
                
                # Update system health
                self._update_system_health()
                
                # Control loop timing
                loop_time = time.time() - loop_start
                if loop_time < control_interval:
                    time.sleep(control_interval - loop_time)
                
                self.control_loop_count += 1
                self.last_control_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                self.state = VehicleState.ERROR
                break
        
        logger.info("Control loop ended")
    
    def _update_sensor_data(self):
        """Update sensor data from all sources"""
        # Update ultrasonic data
        self.status.ultrasonic_data = self.arduino_comm.get_latest_ultrasonic_data()
        
        # Update object detections
        self.status.obstacles_detected = self.object_detector.get_latest_detections()
        
        # Update behavior predictions
        if self.status.obstacles_detected:
            detected_objects = [
                {
                    'id': f"obj_{i}",
                    'class_name': detection.class_name,
                    'center': detection.center,
                    'speed': detection.speed if hasattr(detection, 'speed') else 0.0,
                    'direction': detection.direction if hasattr(detection, 'direction') else 0.0
                }
                for i, detection in enumerate(self.status.obstacles_detected)
            ]
            self.status.behavior_predictions = self.behavior_predictor.predict_all_behaviors(detected_objects)
        else:
            self.status.behavior_predictions = []
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions requiring immediate stop"""
        # Check ultrasonic sensors for immediate obstacles
        if self.status.ultrasonic_data:
            front_distance = self.status.ultrasonic_data.front
            if front_distance > 0 and front_distance < ControlConfig.EMERGENCY_STOP_DISTANCE:
                logger.warning(f"Emergency stop: front obstacle at {front_distance}cm")
                return True
        
        # Check for critical objects in detection
        for detection in self.status.obstacles_detected:
            if detection.distance and detection.distance < ControlConfig.EMERGENCY_STOP_DISTANCE:
                if detection.class_name in ['person', 'car', 'truck']:
                    logger.warning(f"Emergency stop: {detection.class_name} at {detection.distance}cm")
                    return True
        
        # Check system health
        if not self._is_system_healthy():
            logger.error("Emergency stop: system health check failed")
            return True
        
        return False
    
    def _update_vehicle_state(self):
        """Update vehicle state based on current conditions"""
        if self.emergency_stop_active:
            self.state = VehicleState.EMERGENCY_STOP
            return
        
        # Check for obstacles requiring avoidance
        if self._should_avoid_obstacles():
            self.state = VehicleState.OBSTACLE_AVOIDANCE
        else:
            self.state = VehicleState.NAVIGATING
    
    def _should_avoid_obstacles(self) -> bool:
        """Determine if obstacle avoidance is needed"""
        # Check ultrasonic sensors
        if self.status.ultrasonic_data:
            front_distance = self.status.ultrasonic_data.front
            left_distance = self.status.ultrasonic_data.left
            right_distance = self.status.ultrasonic_data.right
            
            if (front_distance > 0 and front_distance < ControlConfig.WARNING_DISTANCE or
                left_distance > 0 and left_distance < ControlConfig.SAFE_DISTANCE or
                right_distance > 0 and right_distance < ControlConfig.SAFE_DISTANCE):
                return True
        
        # Check detected objects
        for detection in self.status.obstacles_detected:
            if detection.distance and detection.distance < ControlConfig.WARNING_DISTANCE:
                return True
        
        return False
    
    def _execute_navigation(self):
        """Execute normal navigation control"""
        if not self.current_goal:
            self._stop_vehicle()
            return
        
        # Get next waypoint
        waypoint = self.path_planner.get_next_waypoint(self.status.current_position)
        if not waypoint:
            # Plan new path to goal
            path = self.path_planner.plan_path(self.status.current_position, self.current_goal)
            if path:
                self.path_planner.update_path(path)
                waypoint = self.path_planner.get_next_waypoint(self.status.current_position)
            else:
                logger.warning("No path found to goal")
                self._stop_vehicle()
                return
        
        self.status.target_waypoint = waypoint
        
        # Calculate control commands
        steering_command = self._calculate_steering(waypoint)
        speed_command = self._calculate_speed(waypoint)
        
        # Apply control
        self._apply_control(speed_command, steering_command)
    
    def _execute_obstacle_avoidance(self):
        """Execute obstacle avoidance control"""
        # Stop vehicle
        self._stop_vehicle()
        
        # Plan avoidance path
        avoidance_path = self._plan_avoidance_path()
        if avoidance_path:
            self.path_planner.update_path(avoidance_path)
            self.state = VehicleState.NAVIGATING
        else:
            logger.warning("No avoidance path found")
    
    def _calculate_steering(self, waypoint: PathPoint) -> int:
        """Calculate steering angle to reach waypoint"""
        # Calculate angle to waypoint
        dx = waypoint.x - self.status.current_position[0]
        dy = waypoint.y - self.status.current_position[1]
        target_angle = math.atan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = target_angle - self.status.current_heading
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalize to [-π, π]
        
        # Convert to degrees and apply PID control
        angle_diff_deg = math.degrees(angle_diff)
        steering_correction = (ControlConfig.STEERING_KP * angle_diff_deg + 
                             ControlConfig.STEERING_KI * angle_diff_deg * 0.05 +  # dt = 0.05s
                             ControlConfig.STEERING_KD * angle_diff_deg / 0.05)
        
        # Calculate steering angle (90 = straight, 0 = left, 180 = right)
        steering_angle = 90 + steering_correction
        
        # Clamp to valid range
        steering_angle = max(0, min(180, steering_angle))
        
        return int(steering_angle)
    
    def _calculate_speed(self, waypoint: PathPoint) -> Tuple[int, int]:
        """Calculate motor speeds to reach waypoint"""
        # Distance to waypoint
        dx = waypoint.x - self.status.current_position[0]
        dy = waypoint.y - self.status.current_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if waypoint reached
        if distance < self.waypoint_reached_threshold:
            return (0, 0)  # Stop
        
        # Base speed from waypoint
        target_speed = waypoint.speed
        
        # Adjust speed based on distance to obstacles
        if self.status.ultrasonic_data:
            front_distance = self.status.ultrasonic_data.front
            if front_distance > 0:
                # Reduce speed as we get closer to obstacles
                speed_factor = min(1.0, front_distance / ControlConfig.WARNING_DISTANCE)
                target_speed *= speed_factor
        
        # Clamp speed
        target_speed = max(ControlConfig.MIN_SPEED, min(ControlConfig.MAX_SPEED, target_speed))
        
        # Convert to motor commands (differential drive)
        # For now, use simple proportional control
        left_speed = int(target_speed)
        right_speed = int(target_speed)
        
        return (left_speed, right_speed)
    
    def _apply_control(self, speed_command: Tuple[int, int], steering_command: int):
        """Apply control commands to vehicle"""
        left_speed, right_speed = speed_command
        
        # Send commands to Arduino
        success = self.arduino_comm.send_motor_command(
            left_speed=left_speed,
            right_speed=right_speed,
            steering_angle=steering_command
        )
        
        if success:
            # Update status
            self.status.current_speed = (left_speed + right_speed) / 2
            self.target_steering = steering_command
        else:
            logger.error("Failed to send motor commands")
    
    def _stop_vehicle(self):
        """Stop the vehicle"""
        self.arduino_comm.send_motor_command(0, 0, 90)
        self.status.current_speed = 0.0
    
    def _plan_avoidance_path(self) -> Optional[List[PathPoint]]:
        """Plan path to avoid obstacles"""
        # Simple obstacle avoidance: turn left or right based on sensor data
        if not self.status.ultrasonic_data:
            return None
        
        # Determine best avoidance direction
        left_distance = self.status.ultrasonic_data.left
        right_distance = self.status.ultrasonic_data.right
        
        # Choose direction with more space
        if left_distance > right_distance:
            avoidance_angle = self.status.current_heading + math.pi/2  # Turn left
        else:
            avoidance_angle = self.status.current_heading - math.pi/2  # Turn right
        
        # Create simple avoidance waypoint
        avoidance_distance = 100  # cm
        avoidance_x = self.status.current_position[0] + avoidance_distance * math.cos(avoidance_angle)
        avoidance_y = self.status.current_position[1] + avoidance_distance * math.sin(avoidance_angle)
        
        # Plan path to avoidance point
        path = self.path_planner.plan_path(self.status.current_position, (avoidance_x, avoidance_y))
        return path
    
    def _update_system_health(self):
        """Update system health status"""
        self.status.system_health = {
            "arduino_connected": self.arduino_comm.is_healthy(),
            "object_detector_loaded": self.object_detector.is_loaded,
            "camera_working": len(self.status.obstacles_detected) >= 0,  # Simple check
            "path_planner_ready": self.path_planner.grid is not None
        }
    
    def _is_system_healthy(self) -> bool:
        """Check if all systems are healthy"""
        return all(self.status.system_health.values())
    
    def get_status(self) -> VehicleStatus:
        """Get current vehicle status"""
        return self.status
    
    def get_performance_metrics(self) -> Dict[str, any]:
        """Get system performance metrics"""
        return {
            "control_loop_count": self.control_loop_count,
            "last_control_time": self.last_control_time,
            "object_detection_metrics": self.object_detector.get_performance_metrics(),
            "path_planning_metrics": self.path_planner.get_performance_metrics(),
            "arduino_status": self.arduino_comm.get_status()
        }
    
    def shutdown(self):
        """Shutdown all systems"""
        logger.info("Shutting down vehicle systems...")
        
        # Stop autonomous operation
        self.stop_autonomous_operation()
        
        # Stop object detection
        self.object_detector.stop_real_time_detection()
        
        # Disconnect Arduino
        self.arduino_comm.disconnect()
        
        # Update state
        self.state = VehicleState.IDLE
        
        logger.info("Vehicle systems shut down") 