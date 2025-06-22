# AI-Based Autonomous Vehicle - User Manual

## Table of Contents
1. [System Overview](#system-overview)
2. [Hardware Setup](#hardware-setup)
3. [Software Installation](#software-installation)
4. [Configuration](#configuration)
5. [Operation](#operation)
6. [Troubleshooting](#troubleshooting)
7. [Safety Guidelines](#safety-guidelines)
8. [Maintenance](#maintenance)

## System Overview

The AI-Based Autonomous Vehicle system is a complete self-driving car prototype that combines:

- **Raspberry Pi 4**: Main computing platform running AI algorithms
- **Arduino**: Hardware interface for motor control and sensor reading
- **YOLO Object Detection**: Real-time identification of obstacles and objects
- **A* Path Planning**: Intelligent navigation and obstacle avoidance
- **Multi-sensor Fusion**: Camera, ultrasonic sensors, and optional LiDAR

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS VEHICLE                       │
├─────────────────────────────────────────────────────────────┤
│  Raspberry Pi 4    │    Arduino Uno    │    Sensors        │
│  ┌─────────────┐   │   ┌─────────────┐ │  ┌─────────────┐  │
│  │ • YOLO AI   │   │   │ • Motor     │ │  │ • Camera    │  │
│  │ • Path Plan │◄──┼──►│ • Servo     │ │  │ • Ultrasonic│  │
│  │ • Control   │   │   │ • Sensors   │ │  │ • LiDAR     │  │
│  └─────────────┘   │   └─────────────┘ │  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Hardware Setup

### Required Components

#### Raspberry Pi Setup
- **Raspberry Pi 4** (4GB RAM recommended)
- **32GB+ MicroSD Card** (Class 10 or higher)
- **5V/3A Power Supply**
- **USB Camera** (720p or higher resolution)
- **USB WiFi Adapter** (if not built-in)

#### Arduino Setup
- **Arduino Uno** or **Arduino Nano**
- **6x HC-SR04 Ultrasonic Sensors**
- **4x DC Motors** (12V, 100RPM recommended)
- **2x L298N Motor Drivers**
- **1x SG90 Servo Motor** (for steering)
- **Breadboard** and connecting wires
- **9V Battery** for Arduino power

#### Optional Components
- **RPLIDAR A1** or similar LiDAR sensor
- **MPU6050** IMU sensor
- **NEO-6M** GPS module
- **LCD Display** for status monitoring

#### Arduino Pin Connections

```
Arduino Pin    Component
────────────   ─────────
Pin 2          Ultrasonic Front Echo
Pin 3          Ultrasonic Front Trig
Pin 4          Ultrasonic Right Echo
Pin 5          Ultrasonic Left Echo
Pin 6          Ultrasonic Right Trig
Pin 7          Ultrasonic Left Trig
Pin 8          Motor Left Forward
Pin 9          Motor Left Backward
Pin 10         Motor Right Forward
Pin 11         Motor Right Backward
Pin 12         Servo Steering
5V             Servo Power
GND            Common Ground
```

#### Motor Driver Connections

```
L298N Driver   Motor
────────────   ─────
IN1            Arduino Pin 8 (Left Forward)
IN2            Arduino Pin 9 (Left Backward)
IN3            Arduino Pin 10 (Right Forward)
IN4            Arduino Pin 11 (Right Backward)
ENA            PWM Pin (Left Speed)
ENB            PWM Pin (Right Speed)
12V            Motor Power Supply
GND            Common Ground
```

### Assembly Instructions

1. **Mount Components**
   - Secure Raspberry Pi to the vehicle chassis
   - Mount Arduino and breadboard
   - Position ultrasonic sensors (front, left, right)
   - Mount camera at the front of the vehicle
   - Secure motors and wheels

2. **Connect Power**
   - Connect 5V power supply to Raspberry Pi
   - Connect 9V battery to Arduino
   - Connect 12V power supply to motor drivers

3. **Wire Sensors**
   - Connect ultrasonic sensors to Arduino pins
   - Connect camera to Raspberry Pi USB port
   - Connect servo to Arduino pin 12

4. **Connect Motors**
   - Wire motors through L298N drivers
   - Connect drivers to Arduino control pins
   - Test motor direction and adjust if needed

## Software Installation

### Automatic Installation

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd aicv
   ```

2. **Run Installation Script**
   ```bash
   chmod +x scripts/install.sh
   ./scripts/install.sh
   ```

3. **Activate Virtual Environment**
   ```bash
   source venv/bin/activate
   ```

### Manual Installation

1. **Install System Dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip python3-venv python3-dev
   sudo apt-get install libatlas-base-dev libhdf5-dev libhdf5-serial-dev
   sudo apt-get install libjasper-dev libqtcore4 libqtgui4 libqt4-test
   sudo apt-get install libgstreamer1.0-0 libgstreamer-plugins-base1.0-0
   sudo apt-get install libgtk-3-0 libavcodec-dev libavformat-dev
   sudo apt-get install libswscale-dev libv4l-dev libxvidcore-dev
   sudo apt-get install libx264-dev libjpeg-dev libpng-dev libtiff-dev
   sudo apt-get install gfortran wget curl git
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO Model**
   ```bash
   mkdir -p models
   wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

5. **Setup Arduino**
   - Install Arduino IDE
   - Install RunningAverage and Servo libraries
   - Upload `scripts/arduino_ultrasonic.ino` to Arduino

6. **Setup Permissions**
   ```bash
   sudo usermod -a -G dialout $USER
   sudo usermod -a -G video $USER
   # Reboot required
   ```

## Configuration

### Hardware Configuration

Edit `config/settings.py` to match your hardware:

```python
class HardwareConfig:
    # Update these values for your setup
    ARDUINO_PORT = "/dev/ttyUSB0"  # Check with: ls /dev/ttyUSB*
    CAMERA_INDEX = 0               # Try 0, 1, 2 if camera not detected
    LIDAR_PORT = "/dev/ttyUSB1"    # Only if using LiDAR
```

### Control Parameters

Adjust vehicle behavior in `config/settings.py`:

```python
class ControlConfig:
    MAX_SPEED = 100        # Maximum speed in cm/s
    SAFE_DISTANCE = 100    # Safe distance to obstacles in cm
    WARNING_DISTANCE = 150 # Warning distance in cm
    EMERGENCY_STOP_DISTANCE = 30  # Emergency stop distance in cm
```

### AI Model Settings

Configure object detection parameters:

```python
class ModelConfig:
    YOLO_CONFIDENCE_THRESHOLD = 0.5  # Detection confidence (0.0-1.0)
    YOLO_NMS_THRESHOLD = 0.4         # Non-maximum suppression
    GRID_SIZE = 50                   # Path planning grid size in cm
```

## Operation

### Basic Operation

1. **Start the System**
   ```bash
   source venv/bin/activate
   python main.py
   ```

2. **Set Navigation Goal**
   ```bash
   python main.py --goal 100 200  # Navigate to (100cm, 200cm)
   ```

3. **Run Demonstration**
   ```bash
   python main.py --demo
   ```

4. **Run System Tests**
   ```bash
   python main.py --test
   ```

### Advanced Operation

1. **Custom Port Configuration**
   ```bash
   python main.py --port /dev/ttyACM0 --camera 1
   ```

2. **Performance Monitoring**
   ```bash
   # Monitor logs in real-time
   tail -f logs/autonomous_vehicle.log
   
   # Check performance metrics
   tail -f logs/performance.log
   ```

3. **Export Logs**
   ```bash
   python -c "from core.logger import logger; logger.export_logs('exports/')"
   ```

### Control Commands

The system supports these control modes:

- **Autonomous Navigation**: Automatic path planning and obstacle avoidance
- **Manual Control**: Direct motor control via commands
- **Emergency Stop**: Immediate halt of all movement
- **Test Mode**: System validation and calibration

## Troubleshooting

### Common Issues and Solutions

#### 1. Arduino Connection Failed

**Symptoms**: "Failed to connect to Arduino" error

**Solutions**:
```bash
# Check available ports
ls /dev/ttyUSB*
ls /dev/ttyACM*


```

#### 2. Camera Not Working

**Symptoms**: "Failed to open camera" error

**Solutions**:
```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Try different camera index
python main.py --camera 1

# Check camera permissions
ls -l /dev/video*
```

#### 3. YOLO Model Issues

**Symptoms**: "Model not loaded" error

**Solutions**:
```bash
# Download model manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
mv yolov8n.pt models/

# Check model file
ls -la models/yolov8n.pt
```

#### 4. Permission Issues

**Symptoms**: "Permission denied" errors

**Solutions**:
```bash
# Add user to required groups
sudo usermod -a -G dialout $USER
sudo usermod -a -G video $USER

# Reboot system
sudo reboot
```

#### 5. Performance Issues

**Symptoms**: Slow object detection or path planning

**Solutions**:
```bash

htop
free -h

```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set debug level
export LOG_LEVEL=DEBUG

# Run with debug output
python main.py

# Check debug logs
tail -f logs/autonomous_vehicle.log
```

### System Diagnostics

Run comprehensive system tests:

```bash
# Run all tests
python scripts/test_system.py

# Test individual components
python -c "from core.arduino_communication import ArduinoCommunication; a = ArduinoCommunication(); print(a.connect())"
```


---
