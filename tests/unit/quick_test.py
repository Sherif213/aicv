#!/usr/bin/env python3
"""
Quick Test for Autonomous Vehicle System
Tests core functionality without hardware dependencies
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_yolo_model():
    """Test YOLO model loading and inference"""
    print("Testing YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (128, 128, 128)  # Gray background
        
        # Run inference
        results = model(test_image)
        print(f"✓ YOLO model working: {len(results)} results")
        return True
    except Exception as e:
        print(f"✗ YOLO model error: {e}")
        return False

def test_path_planning():
    """Test path planning"""
    print("Testing path planning...")
    try:
        from core.path_planning import PathPlanner
        
        planner = PathPlanner()
        planner.initialize_grid(20, 20)
        
        start = (0.0, 0.0)
        goal = (100.0, 100.0)
        
        path = planner.plan_path(start, goal)
        if path:
            print(f"✓ Path planning working: {len(path)} waypoints")
            return True
        else:
            print("✗ Path planning failed")
            return False
    except Exception as e:
        print(f"✗ Path planning error: {e}")
        return False

def test_vehicle_controller():
    """Test vehicle controller"""
    print("Testing vehicle controller...")
    try:
        from core.vehicle_controller import VehicleController
        
        controller = VehicleController()
        status = controller.get_status()
        print(f"✓ Vehicle controller working: {status.state}")
        return True
    except Exception as e:
        print(f"✗ Vehicle controller error: {e}")
        return False

def test_arduino_communication():
    """Test Arduino communication (without actual connection)"""
    print("Testing Arduino communication...")
    try:
        from core.arduino_communication import ArduinoCommunication
        
        arduino = ArduinoCommunication()
        print(f"✓ Arduino communication initialized: {arduino.port}")
        return True
    except Exception as e:
        print(f"✗ Arduino communication error: {e}")
        return False

def main():
    """Run all quick tests"""
    print("AI-Based Autonomous Vehicle - Quick Test")
    print("=" * 40)
    
    tests = [
        ("YOLO Model", test_yolo_model),
        ("Path Planning", test_path_planning),
        ("Vehicle Controller", test_vehicle_controller),
        ("Arduino Communication", test_arduino_communication)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 40)
    print("QUICK TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is working correctly.")
        print("\nNext steps:")
        print("1. Connect Arduino with ultrasonic sensors")
        print("2. Connect camera")
        print("3. Run: python main.py --demo")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 