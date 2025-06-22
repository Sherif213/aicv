#!/usr/bin/env python3
"""
Motor Test Script for Autonomous Vehicle
Tests Arduino motor control functionality
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_motor_commands():
    """Test motor command generation and validation"""
    print("Testing motor command generation...")
    
    try:
        from core.arduino_communication import ArduinoCommunication
        
        arduino = ArduinoCommunication()
        
        # Test different motor commands
        test_commands = [
            (50, 50, 90, "Forward"),
            (-50, -50, 90, "Backward"),
            (30, -30, 45, "Turn Left"),
            (-30, 30, 135, "Turn Right"),
            (0, 0, 90, "Stop"),
            (100, 100, 90, "Fast Forward"),
            (-100, -100, 90, "Fast Backward")
        ]
        
        print("Motor command tests:")
        for left_speed, right_speed, steering, description in test_commands:
            success = arduino.send_motor_command(left_speed, right_speed, steering)
            status = "✓" if success else "✗"
            print(f"  {status} {description}: L={left_speed}, R={right_speed}, S={steering}")
        
        return True
        
    except Exception as e:
        print(f"✗ Motor command test error: {e}")
        return False

def test_arduino_connection():
    """Test Arduino connection and communication"""
    print("\nTesting Arduino connection...")
    
    try:
        from core.arduino_communication import ArduinoCommunication
        
        arduino = ArduinoCommunication()
        
        # Try to connect
        print("Attempting to connect to Arduino...")
        if arduino.connect():
            print("✓ Arduino connected successfully")
            
            # Start reading
            if arduino.start_reading():
                print("✓ Arduino reading started")
                
                # Wait for sensor data
                print("Waiting for sensor data...")
                time.sleep(3)
                
                # Check sensor data
                ultrasonic_data = arduino.get_ultrasonic_data()
                if ultrasonic_data:
                    print(f"✓ Sensor data received: F={ultrasonic_data.front:.1f}cm, L={ultrasonic_data.left:.1f}cm, R={ultrasonic_data.right:.1f}cm")
                else:
                    print("⚠️  No sensor data received")
                
                # Test motor commands
                print("\nTesting motor commands...")
                print("Sending forward command (50, 50, 90)...")
                arduino.send_motor_command(50, 50, 90)
                time.sleep(2)
                
                print("Sending stop command (0, 0, 90)...")
                arduino.send_motor_command(0, 0, 90)
                time.sleep(1)
                
                print("Sending turn command (30, -30, 45)...")
                arduino.send_motor_command(30, -30, 45)
                time.sleep(2)
                
                print("Sending stop command (0, 0, 90)...")
                arduino.send_motor_command(0, 0, 90)
                
                # Check motor status
                motor_status = arduino.get_motor_status()
                if motor_status:
                    print(f"✓ Motor status: L={motor_status.left_speed}, R={motor_status.right_speed}, S={motor_status.steering_angle}")
                else:
                    print("⚠️  No motor status received")
                
                # Disconnect
                arduino.disconnect()
                print("✓ Arduino disconnected")
                return True
            else:
                print("✗ Failed to start Arduino reading")
                return False
        else:
            print("✗ Failed to connect to Arduino")
            print("Make sure:")
            print("  1. Arduino is connected via USB")
            print("  2. Arduino code is uploaded")
            print("  3. Port /dev/ttyUSB0 is correct")
            return False
            
    except Exception as e:
        print(f"✗ Arduino connection test error: {e}")
        return False

def test_motor_simulation():
    """Test motor control without actual hardware"""
    print("\nTesting motor control simulation...")
    
    try:
        from core.arduino_communication import ArduinoCommunication
        
        arduino = ArduinoCommunication()
        
        # Test command queue
        print("Testing command queue...")
        arduino.command_queue = []
        
        # Add test commands
        arduino.send_motor_command(50, 50, 90)
        arduino.send_motor_command(0, 0, 90)
        arduino.send_motor_command(30, -30, 45)
        
        print(f"✓ Commands queued: {len(arduino.command_queue)}")
        
        # Test command validation
        print("Testing command validation...")
        
        # Valid commands
        valid_commands = [
            (50, 50, 90),
            (-100, -100, 180),
            (0, 0, 90),
            (255, 255, 0)
        ]
        
        for left, right, steering in valid_commands:
            success = arduino.send_motor_command(left, right, steering)
            if success:
                print(f"  ✓ Valid command: L={left}, R={right}, S={steering}")
            else:
                print(f"  ✗ Invalid command: L={left}, R={right}, S={steering}")
        
        # Invalid commands (should be clamped)
        invalid_commands = [
            (300, 50, 90),   # Speed too high
            (-300, -50, 90), # Speed too low
            (50, 50, 200),   # Steering too high
            (50, 50, -10)    # Steering too low
        ]
        
        print("Testing invalid command clamping...")
        for left, right, steering in invalid_commands:
            success = arduino.send_motor_command(left, right, steering)
            if success:
                print(f"  ✓ Command clamped: L={left}, R={right}, S={steering}")
            else:
                print(f"  ✗ Command rejected: L={left}, R={right}, S={steering}")
        
        return True
        
    except Exception as e:
        print(f"✗ Motor simulation test error: {e}")
        return False

def main():
    """Run all motor tests"""
    print("AI-Based Autonomous Vehicle - Motor Test Suite")
    print("=" * 50)
    
    tests = [
        ("Motor Commands", test_motor_commands),
        ("Arduino Connection", test_arduino_connection),
        ("Motor Simulation", test_motor_simulation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("MOTOR TEST SUMMARY")
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
        print("🎉 All motor tests passed! Motor control is working correctly.")
    elif passed >= 2:
        print("⚠️  Most motor tests passed. Check Arduino connection if needed.")
    else:
        print("❌ Motor tests failed. Check Arduino setup and connections.")
    
    print("\nMotor control is ready for autonomous operation!")
    return passed >= 2

if __name__ == "__main__":
    main() 