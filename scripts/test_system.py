#!/usr/bin/env python3
"""
System Test Script for AI-Based Autonomous Vehicle
Tests all components and validates system functionality
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.vehicle_controller import VehicleController, VehicleState
from core.arduino_communication import ArduinoCommunication
from models.object_detection import ObjectDetector
from core.path_planning import PathPlanner
from core.logger import logger
from config.settings import HardwareConfig


class SystemTester:
    """Comprehensive system testing for autonomous vehicle"""
    
    def __init__(self):
        self.vehicle = VehicleController()
        self.test_results = {}
        
    def run_all_tests(self) -> bool:
        """Run all system tests"""
        logger.info("Starting comprehensive system tests...")
        
        tests = [
            ("Hardware Connection", self.test_hardware_connection),
            ("Object Detection", self.test_object_detection),
            ("Path Planning", self.test_path_planning),
            ("Motor Control", self.test_motor_control),
            ("Sensor Integration", self.test_sensor_integration),
            ("Emergency Systems", self.test_emergency_systems),
            ("Performance", self.test_performance)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "PASSED" if result else "FAILED"
                logger.info(f"Test {test_name}: {status}")
                
                if not result:
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.test_results[test_name] = False
                all_passed = False
        
        self.print_test_summary()
        return all_passed
    
    def test_hardware_connection(self) -> bool:
        """Test hardware connections"""
        logger.info("Testing hardware connections...")
        
        # Test Arduino connection
        arduino = ArduinoCommunication()
        if not arduino.connect():
            logger.error("Arduino connection failed")
            return False
        
        if not arduino.start_reading():
            logger.error("Arduino reading failed")
            return False
        
        # Wait for sensor data
        time.sleep(2)
        ultrasonic_data = arduino.get_ultrasonic_data()
        if not ultrasonic_data:
            logger.error("No ultrasonic data received")
            return False
        
        logger.info(f"Ultrasonic data: F={ultrasonic_data.front}, L={ultrasonic_data.left}, R={ultrasonic_data.right}")
        
        # Test camera
        cap = cv2.VideoCapture(HardwareConfig.CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("Camera connection failed")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logger.error("Camera frame capture failed")
            return False
        
        logger.info(f"Camera test passed: frame size {frame.shape}")
        
        arduino.disconnect()
        return True
    
    def test_object_detection(self) -> bool:
        """Test object detection system"""
        logger.info("Testing object detection...")
        
        detector = ObjectDetector()
        
        # Test model loading
        if not detector.load_model():
            logger.error("Object detection model loading failed")
            return False
        
        # Test detection on test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (128, 128, 128)  # Gray background
        
        # Add a simple rectangle to simulate an object
        cv2.rectangle(test_image, (200, 150), (400, 350), (255, 0, 0), 2)
        
        detections = detector.detect_objects(test_image)
        logger.info(f"Object detection test: {len(detections)} detections found")
        
        # Test real-time detection
        if not detector.start_real_time_detection():
            logger.error("Real-time detection failed to start")
            return False
        
        time.sleep(3)  # Let it run for a few seconds
        
        latest_detections = detector.get_latest_detections()
        logger.info(f"Real-time detection: {len(latest_detections)} detections")
        
        detector.stop_real_time_detection()
        return True
    
    def test_path_planning(self) -> bool:
        """Test path planning system"""
        logger.info("Testing path planning...")
        
        planner = PathPlanner()
        
        # Initialize grid
        planner.initialize_grid(50, 50)  # 2.5m x 2.5m grid
        
        # Test simple path planning
        start = (0.0, 0.0)
        goal = (200.0, 200.0)
        
        path = planner.plan_path(start, goal)
        if not path:
            logger.error("Path planning failed")
            return False
        
        logger.info(f"Path planning test: {len(path)} waypoints generated")
        
        # Test obstacle avoidance
        obstacles = [(100.0, 100.0, 30.0)]  # Obstacle at center
        planner.update_obstacles(obstacles)
        
        path_with_obstacles = planner.plan_path(start, goal)
        if not path_with_obstacles:
            logger.error("Path planning with obstacles failed")
            return False
        
        logger.info(f"Obstacle avoidance test: {len(path_with_obstacles)} waypoints")
        
        return True
    
    def test_motor_control(self) -> bool:
        """Test motor control system"""
        logger.info("Testing motor control...")
        
        arduino = ArduinoCommunication()
        if not arduino.connect():
            logger.error("Arduino connection failed for motor test")
            return False
        
        if not arduino.start_reading():
            logger.error("Arduino reading failed for motor test")
            return False
        
        # Test forward movement
        logger.info("Testing forward movement...")
        arduino.send_motor_command(50, 50, 90)
        time.sleep(2)
        
        # Test turning
        logger.info("Testing turning...")
        arduino.send_motor_command(30, -30, 45)
        time.sleep(2)
        
        # Test stop
        logger.info("Testing stop...")
        arduino.send_motor_command(0, 0, 90)
        time.sleep(1)
        
        # Test emergency stop
        logger.info("Testing emergency stop...")
        arduino.emergency_stop()
        time.sleep(1)
        
        arduino.disconnect()
        return True
    
    def test_sensor_integration(self) -> bool:
        """Test sensor integration"""
        logger.info("Testing sensor integration...")
        
        arduino = ArduinoCommunication()
        if not arduino.connect():
            logger.error("Arduino connection failed for sensor test")
            return False
        
        if not arduino.start_reading():
            logger.error("Arduino reading failed for sensor test")
            return False
        
        # Collect sensor data for 5 seconds
        sensor_readings = []
        start_time = time.time()
        
        while time.time() - start_time < 5:
            data = arduino.get_ultrasonic_data()
            if data:
                sensor_readings.append(data)
            time.sleep(0.1)
        
        if len(sensor_readings) < 10:
            logger.error("Insufficient sensor readings")
            return False
        
        # Analyze sensor data
        front_distances = [r.front for r in sensor_readings if r.front > 0]
        left_distances = [r.left for r in sensor_readings if r.left > 0]
        right_distances = [r.right for r in sensor_readings if r.right > 0]
        
        logger.info(f"Sensor integration test: {len(sensor_readings)} readings")
        logger.info(f"Front: {len(front_distances)} valid, Left: {len(left_distances)} valid, Right: {len(right_distances)} valid")
        
        arduino.disconnect()
        return len(sensor_readings) >= 10
    
    def test_emergency_systems(self) -> bool:
        """Test emergency systems"""
        logger.info("Testing emergency systems...")
        
        vehicle = VehicleController()
        if not vehicle.initialize():
            logger.error("Vehicle initialization failed for emergency test")
            return False
        
        # Test emergency stop
        logger.info("Testing emergency stop...")
        vehicle.emergency_stop()
        
        status = vehicle.get_status()
        if status.state != VehicleState.EMERGENCY_STOP:
            logger.error("Emergency stop state not set correctly")
            return False
        
        logger.info("Emergency stop test passed")
        
        vehicle.shutdown()
        return True
    
    def test_performance(self) -> bool:
        """Test system performance"""
        logger.info("Testing system performance...")
        
        # Test object detection performance
        detector = ObjectDetector()
        if not detector.load_model():
            logger.error("Model loading failed for performance test")
            return False
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Measure inference time
        start_time = time.time()
        for _ in range(10):
            detector.detect_objects(test_image)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 10
        fps = 1.0 / avg_inference_time
        
        logger.info(f"Object detection performance: {avg_inference_time:.3f}s per frame, {fps:.1f} FPS")
        
        # Test path planning performance
        planner = PathPlanner()
        planner.initialize_grid(100, 100)
        
        start_time = time.time()
        for _ in range(5):
            start = (np.random.randint(0, 500), np.random.randint(0, 500))
            goal = (np.random.randint(0, 500), np.random.randint(0, 500))
            planner.plan_path(start, goal)
        end_time = time.time()
        
        avg_planning_time = (end_time - start_time) / 5
        logger.info(f"Path planning performance: {avg_planning_time:.3f}s per path")
        
        # Performance thresholds
        if avg_inference_time > 0.5:  # 500ms
            logger.warning("Object detection is slow")
        
        if avg_planning_time > 1.0:  # 1 second
            logger.warning("Path planning is slow")
        
        return True
    
    def print_test_summary(self):
        """Print test results summary"""
        logger.info("\n" + "="*50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*50)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! System is ready for operation.")
        else:
            logger.error("❌ Some tests failed. Please check the system.")
        
        logger.info("="*50)


def main():
    """Main test function"""
    logger.info("AI-Based Autonomous Vehicle - System Test")
    logger.info("="*50)
    
    tester = SystemTester()
    
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Testing failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 