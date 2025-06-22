#!/usr/bin/env python3
"""
Simple Car Movement Test
Tests basic motor control to move the car
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Test car movement"""
    print("Testing Car Movement...")
    
    try:
        from core.arduino_communication import ArduinoCommunication
        
        # Initialize Arduino
        arduino = ArduinoCommunication()
        
        # Connect to Arduino
        print("Connecting to Arduino...")
        if not arduino.connect():
            print("Failed to connect to Arduino!")
            return False
        
        # Start reading
        if not arduino.start_reading():
            print("Failed to start Arduino reading!")
            return False
        
        print("✓ Arduino connected and reading")
        
        # Wait for sensor data
        time.sleep(2)
        
        # Test different movements
        movements = [
            ("Forward", 50, 50, 90),
            ("Stop", 0, 0, 90),
            ("Turn Left", 30, -30, 45),
            ("Stop", 0, 0, 90),
            ("Turn Right", -30, 30, 135),
            ("Stop", 0, 0, 90),
            ("Backward", -50, -50, 90),
            ("Stop", 0, 0, 90)
        ]
        
        for name, left, right, steering in movements:
            print(f"\n{name}...")
            print(f"  Left: {left}, Right: {right}, Steering: {steering}")
            
            # Send command
            arduino.send_motor_command(left, right, steering)
            
            # Wait
            time.sleep(2)
        
        print("\n✓ Movement test completed!")
        
        # Disconnect
        arduino.disconnect()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    main() 