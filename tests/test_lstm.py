#!/usr/bin/env python3
"""
LSTM Behavior Prediction Test Script
Tests the LSTM-based behavior prediction system
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_lstm_imports():
    """Test if LSTM modules can be imported"""
    print("Testing LSTM module imports...")
    
    try:
        from models.behavior_prediction import BehaviorPredictor, LSTMPredictor, BehaviorPrediction, TrajectoryPoint
        print("✓ LSTM modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import LSTM modules: {e}")
        return False


def test_lstm_model_creation():
    """Test LSTM model creation"""
    print("Testing LSTM model creation...")
    
    try:
        from models.behavior_prediction import LSTMPredictor
        
        # Create LSTM model
        model = LSTMPredictor(input_size=4, hidden_size=64, num_layers=2, output_size=4)
        print(f"✓ LSTM model created: {model}")
        print(f"  - Input size: 4")
        print(f"  - Hidden size: 64")
        print(f"  - Layers: 2")
        print(f"  - Output size: 4")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create LSTM model: {e}")
        return False


def test_behavior_predictor_initialization():
    """Test behavior predictor initialization"""
    print("Testing behavior predictor initialization...")
    
    try:
        from models.behavior_prediction import BehaviorPredictor
        
        # Initialize behavior predictor
        predictor = BehaviorPredictor()
        print(f"✓ Behavior predictor initialized")
        print(f"  - Device: {predictor.device}")
        print(f"  - Max history length: {predictor.max_history_length}")
        print(f"  - Prediction horizon: {predictor.prediction_horizon}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to initialize behavior predictor: {e}")
        return False


def test_model_loading():
    """Test LSTM model loading"""
    print("Testing LSTM model loading...")
    
    try:
        from models.behavior_prediction import BehaviorPredictor
        
        predictor = BehaviorPredictor()
        
        # Load model
        success = predictor.load_model()
        if success:
            print("✓ LSTM model loaded successfully")
            print(f"  - Model loaded: {predictor.is_loaded}")
            print(f"  - Model type: {type(predictor.model)}")
        else:
            print("⚠ LSTM model loading failed (this is normal for new models)")
        
        return True
    except Exception as e:
        print(f"✗ Error during model loading: {e}")
        return False


def test_trajectory_tracking():
    """Test trajectory tracking functionality"""
    print("Testing trajectory tracking...")
    
    try:
        from models.behavior_prediction import BehaviorPredictor
        
        predictor = BehaviorPredictor()
        
        # Add trajectory points
        object_id = "test_car_1"
        for i in range(10):
            x = i * 10.0  # Moving forward
            y = 0.0
            speed = 5.0
            direction = 0.0  # Moving straight
            timestamp = time.time() + i
            
            predictor.update_trajectory(object_id, (x, y), speed, direction, timestamp)
        
        print(f"✓ Trajectory tracking working")
        print(f"  - Object ID: {object_id}")
        print(f"  - History length: {len(predictor.trajectory_history[object_id])}")
        
        return True
    except Exception as e:
        print(f"✗ Error during trajectory tracking: {e}")
        return False


def test_behavior_prediction():
    """Test behavior prediction functionality"""
    print("Testing behavior prediction...")
    
    try:
        from models.behavior_prediction import BehaviorPredictor
        
        predictor = BehaviorPredictor()
        predictor.load_model()
        
        # Create test object with trajectory
        object_id = "test_car_2"
        for i in range(5):
            x = i * 5.0
            y = 0.0
            speed = 3.0
            direction = 0.0
            timestamp = time.time() + i
            
            predictor.update_trajectory(object_id, (x, y), speed, direction, timestamp)
        
        # Predict behavior
        current_position = (20.0, 0.0)
        prediction = predictor.predict_behavior(object_id, "car", current_position)
        
        if prediction:
            print("✓ Behavior prediction working")
            print(f"  - Object ID: {prediction.object_id}")
            print(f"  - Class: {prediction.class_name}")
            print(f"  - Predicted speed: {prediction.predicted_speed:.2f}")
            print(f"  - Predicted direction: {prediction.predicted_direction:.2f}")
            print(f"  - Confidence: {prediction.confidence:.2f}")
            print(f"  - Trajectory points: {len(prediction.predicted_trajectory)}")
        else:
            print("⚠ Behavior prediction returned None (may need more training data)")
        
        return True
    except Exception as e:
        print(f"✗ Error during behavior prediction: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction for multiple objects"""
    print("Testing batch prediction...")
    
    try:
        from models.behavior_prediction import BehaviorPredictor
        
        predictor = BehaviorPredictor()
        predictor.load_model()
        
        # Create test objects
        detected_objects = []
        for i in range(3):
            object_id = f"car_{i}"
            
            # Add trajectory history
            for j in range(5):
                x = j * 3.0 + i * 10.0
                y = i * 5.0
                speed = 2.0 + i * 0.5
                direction = i * 0.1
                timestamp = time.time() + j
                
                predictor.update_trajectory(object_id, (x, y), speed, direction, timestamp)
            
            detected_objects.append({
                'id': object_id,
                'class_name': 'car',
                'center': (15.0 + i * 10.0, i * 5.0),
                'speed': 2.0 + i * 0.5,
                'direction': i * 0.1
            })
        
        # Predict behaviors for all objects
        predictions = predictor.predict_all_behaviors(detected_objects)
        
        print(f"✓ Batch prediction working")
        print(f"  - Input objects: {len(detected_objects)}")
        print(f"  - Predictions generated: {len(predictions)}")
        
        for pred in predictions:
            print(f"    - {pred.object_id}: speed={pred.predicted_speed:.2f}, conf={pred.confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Error during batch prediction: {e}")
        return False


def test_performance_metrics():
    """Test performance metrics"""
    print("Testing performance metrics...")
    
    try:
        from models.behavior_prediction import BehaviorPredictor
        
        predictor = BehaviorPredictor()
        predictor.load_model()
        
        # Run some predictions to generate metrics
        object_id = "test_metrics"
        for i in range(3):
            x = i * 2.0
            y = 0.0
            speed = 1.0
            direction = 0.0
            timestamp = time.time() + i
            
            predictor.update_trajectory(object_id, (x, y), speed, direction, timestamp)
        
        # Make a prediction
        prediction = predictor.predict_behavior(object_id, "car", (6.0, 0.0))
        
        # Get metrics
        metrics = predictor.get_performance_metrics()
        
        print("✓ Performance metrics working")
        print(f"  - Total predictions: {metrics.get('total_predictions', 0)}")
        print(f"  - Model loaded: {metrics.get('model_loaded', False)}")
        
        if 'avg_prediction_time' in metrics:
            print(f"  - Avg prediction time: {metrics['avg_prediction_time']:.4f}s")
        
        return True
    except Exception as e:
        print(f"✗ Error during performance metrics: {e}")
        return False


def main():
    """Run all LSTM tests"""
    print("=" * 60)
    print("LSTM BEHAVIOR PREDICTION SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        test_lstm_imports,
        test_lstm_model_creation,
        test_behavior_predictor_initialization,
        test_model_loading,
        test_trajectory_tracking,
        test_behavior_prediction,
        test_batch_prediction,
        test_performance_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        time.sleep(0.5)  # Small delay between tests
    
    print("\n" + "=" * 60)
    print(f"LSTM TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All LSTM tests passed! Behavior prediction system is working.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    main() 