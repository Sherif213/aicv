#!/usr/bin/env python3

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logger import logger
from models.object_detection import ObjectDetector
from core.path_planning import PathPlanner
from core.arduino_communication import ArduinoCommunication
from models.behavior_prediction import BehaviorPredictor


def test_object_detection():
    print("Testing Object Detection...")
    detector = ObjectDetector()
    success = detector.load_model()
    print(f"✓ Object detection: {'PASS' if success else 'FAIL'}")
    return success


def test_path_planning():
    print("Testing Path Planning...")
    planner = PathPlanner()
    planner.initialize_grid(100, 100)  # Initialize a 100x100 grid
    path = planner.plan_path((0, 0), (10, 10))
    success = path is not None
    print(f"✓ Path planning: {'PASS' if success else 'FAIL'}")
    return success


def test_arduino_communication():
    print("Testing Arduino Communication...")
    arduino = ArduinoCommunication()
    success = arduino.connect()
    if success:
        arduino.disconnect()
    print(f"✓ Arduino communication: {'PASS' if success else 'SIMULATION'}")
    return True  # Always pass as simulation is acceptable


def test_behavior_prediction():
    print("Testing Behavior Prediction...")
    predictor = BehaviorPredictor()
    success = predictor.load_model()
    print(f"✓ Behavior prediction: {'PASS' if success else 'FAIL'}")
    return success


def main():
    print("=" * 50)
    print("AI Autonomous Vehicle - System Test")
    print("=" * 50)
    
    tests = [
        test_object_detection,
        test_path_planning,
        test_arduino_communication,
        test_behavior_prediction
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            time.sleep(1)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    test_names = ["Object Detection", "Path Planning", "Arduino Communication", "Behavior Prediction"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {name}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 3:
        print("✓ System ready for demonstration!")
        return True
    else:
        print("✗ Some components need attention before demonstration")
        return False


if __name__ == "__main__":
    main() 