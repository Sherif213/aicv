#!/usr/bin/env python3
"""
AI-Based Autonomous Vehicle - Main Application
Main entry point for the autonomous vehicle system
"""

import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.vehicle_controller import VehicleController, VehicleState
from core.logger import logger
from config.settings import HardwareConfig


class AutonomousVehicle:
    """Main autonomous vehicle application"""
    
    def __init__(self):
        self.controller = VehicleController()
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Autonomous vehicle application initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Initialize the autonomous vehicle system"""
        logger.info("Initializing autonomous vehicle system...")
        
        try:
            # Initialize vehicle controller
            if not self.controller.initialize():
                logger.error("Failed to initialize vehicle controller")
                return False
            
            logger.info("Autonomous vehicle system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during system initialization: {e}")
            return False
    
    def run(self, goal: Optional[Tuple[float, float]] = None, test_mode: bool = False):
        """Run the autonomous vehicle system"""
        if not self.initialize():
            logger.error("Failed to initialize system")
            return False
        
        self.running = True
        logger.info("Starting autonomous vehicle operation...")
        
        try:
            # Set goal if provided
            if goal:
                self.controller.set_goal(goal)
                logger.info(f"Navigation goal set: {goal}")
            
            # Start autonomous operation
            if not self.controller.start_autonomous_operation():
                logger.error("Failed to start autonomous operation")
                return False
            
            # Main application loop
            while self.running:
                try:
                    # Get current status
                    status = self.controller.get_status()
                    
                    # Log status periodically
                    if status.state != VehicleState.IDLE:
                        logger.log_system_status({
                            "state": status.state.value,
                            "position": status.current_position,
                            "speed": status.current_speed,
                            "obstacles": len(status.obstacles_detected),
                            "system_health": status.system_health
                        })
                    
                    # Check for completion in test mode
                    if test_mode and status.state == VehicleState.IDLE:
                        logger.info("Test completed")
                        break
                    
                    # Check for errors
                    if status.state == VehicleState.ERROR:
                        logger.error("System entered error state")
                        break
                    
                    time.sleep(1)  # Status update interval
                    
                except KeyboardInterrupt:
                    logger.info("User interrupted operation")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Error during operation: {e}")
            return False
        
        finally:
            self.shutdown()
        
        return True
    
    def shutdown(self):
        """Shutdown the autonomous vehicle system"""
        logger.info("Shutting down autonomous vehicle system...")
        
        self.running = False
        
        # Shutdown vehicle controller
        if self.controller:
            self.controller.shutdown()
        
        logger.info("Autonomous vehicle system shut down")


def run_demo():
    """Run a demonstration of the autonomous vehicle system"""
    logger.info("Starting autonomous vehicle demonstration...")
    
    vehicle = AutonomousVehicle()
    
    # Demo goals (in cm)
    demo_goals = [
        (100, 0),    # Move forward 1m
        (100, 100),  # Move diagonally
        (0, 100),    # Move left
        (0, 0)       # Return to start
    ]
    
    for i, goal in enumerate(demo_goals):
        logger.info(f"Demo step {i+1}: Navigating to {goal}")
        
        if not vehicle.run(goal=goal, test_mode=True):
            logger.error(f"Demo step {i+1} failed")
            break
        
        time.sleep(2)  # Pause between goals
    
    logger.info("Demonstration completed")


def run_test():
    """Run system tests"""
    logger.info("Running system tests...")
    
    vehicle = AutonomousVehicle()
    
    # Test initialization
    logger.info("Testing system initialization...")
    if not vehicle.initialize():
        logger.error("Initialization test failed")
        return False
    
    # Test object detection
    logger.info("Testing object detection...")
    detector = vehicle.controller.object_detector
    if not detector.is_loaded:
        logger.error("Object detection test failed")
        return False
    
    # Test path planning
    logger.info("Testing path planning...")
    planner = vehicle.controller.path_planner
    if planner.grid is None:
        logger.error("Path planning test failed")
        return False
    
    # Test Arduino communication
    logger.info("Testing Arduino communication...")
    arduino = vehicle.controller.arduino_comm
    if not arduino.is_connected:
        logger.error("Arduino communication test failed")
        return False
    
    logger.info("All system tests passed")
    return True


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="AI-Based Autonomous Vehicle")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--test", action="store_true", help="Run system tests")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), 
                       help="Set navigation goal (x, y in cm)")
    parser.add_argument("--port", type=str, default=HardwareConfig.ARDUINO_PORT,
                       help="Arduino serial port")
    parser.add_argument("--camera", type=int, default=HardwareConfig.CAMERA_INDEX,
                       help="Camera device index")
    
    args = parser.parse_args()
    
    # Update configuration based on arguments
    if args.port:
        HardwareConfig.ARDUINO_PORT = args.port
    if args.camera is not None:
        HardwareConfig.CAMERA_INDEX = args.camera
    
    logger.info("AI-Based Autonomous Vehicle System")
    logger.info("==================================")
    logger.info(f"Arduino Port: {HardwareConfig.ARDUINO_PORT}")
    logger.info(f"Camera Index: {HardwareConfig.CAMERA_INDEX}")
    
    try:
        if args.test:
            # Run system tests
            success = run_test()
            sys.exit(0 if success else 1)
        
        elif args.demo:
            # Run demonstration
            run_demo()
        
        else:
            # Run normal operation
            vehicle = AutonomousVehicle()
            
            goal: Optional[Tuple[float, float]] = None
            if args.goal:
                goal = tuple(args.goal)
            
            success = vehicle.run(goal=goal)
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
