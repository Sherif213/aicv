#!/usr/bin/env python3
"""
Simple Test Script for AI-Based Autonomous Vehicle
Tests core functionality without hardware dependencies
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from config.settings import HardwareConfig, ModelConfig, ControlConfig
        print("✓ Config modules imported")
        
        from core.logger import logger
        print("✓ Logger module imported")
        
        from core.path_planning import PathPlanner
        print("✓ Path planning module imported")
        
        from models.object_detection import ObjectDetector
        print("✓ Object detection module imported")
        
        from core.arduino_communication import ArduinoCommunication
        print("✓ Arduino communication module imported")
        
        from core.vehicle_controller import VehicleController
        print("✓ Vehicle controller module imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_path_planning():
    """Test path planning functionality"""
    print("\nTesting path planning...")
    
    try:
        from core.path_planning import PathPlanner
        
        planner = PathPlanner()
        planner.initialize_grid(20, 20)  # 1m x 1m grid
        
        # Test simple path
        start = (0.0, 0.0)
        goal = (100.0, 100.0)
        
        path = planner.plan_path(start, goal)
        if path:
            print(f"✓ Path planning works: {len(path)} waypoints")
            return True
        else:
            print("✗ Path planning failed")
            return False
            
    except Exception as e:
        print(f"✗ Path planning error: {e}")
        return False

def test_object_detection():
    """Test object detection (without camera)"""
    print("\nTesting object detection...")
    
    try:
        from models.object_detection import ObjectDetector
        
        detector = ObjectDetector()
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (128, 128, 128)  # Gray background
        
        # Test detection without model loading
        detections = detector.detect_objects(test_image)
        print(f"✓ Object detection initialized: {len(detections)} detections")
        return True
        
    except Exception as e:
        print(f"✗ Object detection error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config.settings import HardwareConfig, ModelConfig, ControlConfig
        
        print(f"✓ Hardware config loaded: Arduino port = {HardwareConfig.ARDUINO_PORT}")
        print(f"✓ Model config loaded: Grid size = {ModelConfig.GRID_SIZE}")
        print(f"✓ Control config loaded: Max speed = {ControlConfig.MAX_SPEED}")
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_logger():
    """Test logging functionality"""
    print("\nTesting logger...")
    
    try:
        from core.logger import logger
        
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        print("✓ Logger working")
        return True
        
    except Exception as e:
        print(f"✗ Logger error: {e}")
        return False

def test_vehicle_controller():
    """Test vehicle controller initialization"""
    print("\nTesting vehicle controller...")
    
    try:
        from core.vehicle_controller import VehicleController
        
        controller = VehicleController()
        print("✓ Vehicle controller created")
        
        # Test status
        status = controller.get_status()
        print(f"✓ Controller status: {status.state}")
        return True
        
    except Exception as e:
        print(f"✗ Vehicle controller error: {e}")
        return False

def main():
    """Run all tests"""
    print("AI-Based Autonomous Vehicle - Simple Test Suite")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Logger", test_logger),
        ("Path Planning", test_path_planning),
        ("Object Detection", test_object_detection),
        ("Vehicle Controller", test_vehicle_controller)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready for hardware testing.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 