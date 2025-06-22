# Raspberry Pi Setup Guide for AI-Based Autonomous Vehicle

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [Initial Raspberry Pi Setup](#initial-raspberry-pi-setup)
3. [Operating System Installation](#operating-system-installation)
4. [System Configuration](#system-configuration)
5. [Development Environment Setup](#development-environment-setup)
6. [Hardware Interface Configuration](#hardware-interface-configuration)
7. [AI/ML Framework Installation](#aiml-framework-installation)
8. [Autonomous Vehicle Software Installation](#autonomous-vehicle-software-installation)
9. [Testing and Validation](#testing-and-validation)
10. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Raspberry Pi Components
- **Raspberry Pi 4 Model B** (4GB RAM recommended)
- **MicroSD Card** (32GB Class 10 or higher)
- **Power Supply** (5V/3A USB-C)
- **Micro HDMI Cable** (for initial setup)
- **USB Keyboard and Mouse** (for initial setup)
- **Monitor/Display** (for initial setup)

### Autonomous Vehicle Hardware
- **Arduino Uno/Mega** (for motor control and sensors)
- **Camera Module** (Raspberry Pi Camera v2 or USB webcam)
- **Ultrasonic Sensors** (HC-SR04 x3)
- **DC Motors** (2x for differential drive)
- **Motor Driver** (L298N or similar)
- **Servo Motor** (for steering if applicable)
- **Battery Pack** (12V for motors, 5V for Pi)
- **Chassis and Wheels**
- **Breadboard and Jumper Wires**

---

## Initial Raspberry Pi Setup

### Step 1: Download Raspberry Pi Imager
1. Visit [raspberrypi.org/software](https://www.raspberrypi.org/software/)
2. Download Raspberry Pi Imager for your operating system
3. Install the application

### Step 2: Prepare MicroSD Card
1. Insert microSD card into your computer
2. Open Raspberry Pi Imager
3. Click "Choose OS" → "Raspberry Pi OS (32-bit)" → "Raspberry Pi OS with desktop"
4. Click "Choose Storage" → Select your microSD card
5. Click "Write" and wait for completion

### Step 3: Configure Boot Settings
1. After writing, click "Continue" to eject the card
2. Re-insert the card
3. Navigate to the `boot` partition
4. Create an empty file named `ssh` (enables SSH)
5. Create `wpa_supplicant.conf` for WiFi configuration:

```bash
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="YOUR_WIFI_NAME"
    psk="YOUR_WIFI_PASSWORD"
    key_mgmt=WPA-PSK
}
```

---

## Operating System Installation

### Step 1: First Boot
1. Insert microSD card into Raspberry Pi
2. Connect power supply, monitor, keyboard, and mouse
3. Power on the Raspberry Pi
4. Wait for first boot (may take 5-10 minutes)

### Step 2: Initial Configuration
1. **Country/Language Setup:**
   - Select your country
   - Choose keyboard layout
   - Set timezone

2. **User Account Setup:**
   - Create username: `pi` (or custom)
   - Set password: `raspberry` (change for security)
   - Enable auto-login (optional)

3. **Display Configuration:**
   - Set screen resolution
   - Enable overscan if needed

4. **Advanced Options:**
   - Expand filesystem
   - Enable SSH
   - Set GPU memory split (128MB recommended)

### Step 3: System Update
```bash
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
```

---

## System Configuration

### Step 1: Enable Required Interfaces
```bash
sudo raspi-config
```

Navigate to:
- **Interface Options** → **Camera** → **Enable**
- **Interface Options** → **SSH** → **Enable**
- **Interface Options** → **Serial Port** → **Enable**
- **Interface Options** → **I2C** → **Enable**
- **Interface Options** → **SPI** → **Enable**

### Step 2: Configure Serial Port for Arduino
```bash
sudo nano /boot/config.txt
```

Add these lines:
```
# Enable UART
enable_uart=1
dtoverlay=disable-bt
```

### Step 3: Set Up User Permissions
```bash
# Add user to required groups
sudo usermod -a -G gpio,i2c,spi,video $USER

# Create udev rules for Arduino
sudo nano /etc/udev/rules.d/99-arduino.rules
```

Add this content:
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0043", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0001", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0243", MODE="0666"
```

### Step 4: Reboot System
```bash
sudo reboot
```

---

## Development Environment Setup

### Step 1: Install Python and Development Tools
```bash
# Install Python 3 and pip
sudo apt install python3 python3-pip python3-venv -y

# Install development tools
sudo apt install git build-essential cmake pkg-config -y

# Install image processing libraries
sudo apt install libjpeg-dev libpng-dev libtiff-dev -y
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt install libxvidcore-dev libx264-dev -y

# Install GUI libraries
sudo apt install libgtk-3-dev -y

# Install mathematical libraries
sudo apt install libatlas-base-dev gfortran -y
```

### Step 2: Install OpenCV Dependencies
```bash
# Install OpenCV system dependencies
sudo apt install libopencv-dev python3-opencv -y

# Verify OpenCV installation
python3 -c "import cv2; print(cv2.__version__)"
```

### Step 3: Set Up Virtual Environment
```bash
# Create project directory
mkdir ~/aicv
cd ~/aicv

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## Hardware Interface Configuration

### Step 1: Camera Setup
```bash
# Test camera
vcgencmd get_camera
# Should return: supported=1 detected=1

# Test camera capture
raspistill -o test.jpg
```

### Step 2: Arduino Connection Test
```bash
# Install pyserial
pip install pyserial

# Test Arduino connection
python3 -c "
import serial
import glob
ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
print('Available ports:', ports)
"
```

### Step 3: GPIO Configuration
```bash
# Install GPIO library
pip install RPi.GPIO

# Test GPIO
python3 -c "
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
print('GPIO library working')
"
```

---

## AI/ML Framework Installation

### Step 1: Install PyTorch
```bash
# Install PyTorch for ARM (CPU version for Raspberry Pi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install Computer Vision Libraries
```bash
# Install OpenCV for Python
pip install opencv-python

# Install image processing libraries
pip install pillow matplotlib

# Install scientific computing libraries
pip install numpy scipy
```

### Step 3: Install YOLO and Ultralytics
```bash
# Install Ultralytics (YOLOv8)
pip install ultralytics

# Download YOLO model
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('YOLO model downloaded successfully')
"
```

### Step 4: Install Additional ML Libraries
```bash
# Install scikit-learn for machine learning
pip install scikit-learn

# Install pandas for data manipulation
pip install pandas

# Install additional utilities
pip install tqdm psutil
```

---

## Autonomous Vehicle Software Installation

### Step 1: Clone Project Repository
```bash
# Navigate to project directory
cd ~/aicv

# Clone repository (if using git)
# git clone <your-repository-url> .

# Or create project structure manually
mkdir -p {config,core,models,scripts,logs,data}
```

### Step 2: Install Project Dependencies
```bash
# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
pyserial>=3.5
RPi.GPIO>=0.7.0
tqdm>=4.65.0
psutil>=5.9.0
Pillow>=10.0.0
pyyaml>=6.0
requests>=2.31.0
EOF

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Project Settings
```bash
# Create configuration files
mkdir -p config
cat > config/settings.py << 'EOF'
"""
Configuration settings for AI-Based Autonomous Vehicle
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class HardwareConfig:
    """Hardware configuration"""
    # Camera settings
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Arduino communication
    ARDUINO_PORT = "/dev/ttyUSB0"
    ARDUINO_BAUDRATE = 115200
    
    # GPIO pins
    MOTOR_LEFT_FORWARD = 17
    MOTOR_LEFT_BACKWARD = 18
    MOTOR_RIGHT_FORWARD = 27
    MOTOR_RIGHT_BACKWARD = 22
    SERVO_PIN = 23

@dataclass
class ModelConfig:
    """AI model configuration"""
    YOLO_MODEL_PATH = "models/yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    LSTM_SEQUENCE_LENGTH = 10
    PREDICTION_HORIZON = 5

@dataclass
class ControlConfig:
    """Control system configuration"""
    CONTROL_FREQUENCY = 20  # Hz
    MAX_SPEED = 100
    MIN_SPEED = 0
    STEERING_CENTER = 90
    STEERING_RANGE = 45

@dataclass
class SensorConfig:
    """Sensor configuration"""
    ULTRASONIC_TRIGGER_PIN = 24
    ULTRASONIC_ECHO_PIN = 25
    SAFE_DISTANCE = 30  # cm

@dataclass
class CommunicationConfig:
    """Communication configuration"""
    SERIAL_TIMEOUT = 1.0
    HEARTBEAT_INTERVAL = 0.1

@dataclass
class LoggingConfig:
    """Logging configuration"""
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = Path("logs/autonomous_vehicle.log")
    PERFORMANCE_LOG_FILE = Path("logs/performance.log")
    DETECTION_LOG_FILE = Path("logs/detection.log")
    SENSOR_LOG_FILE = Path("logs/sensor.log")

# Create log directories
Path("logs").mkdir(exist_ok=True)
EOF
```

### Step 4: Set Up Arduino Firmware
```bash
# Create Arduino sketch directory
mkdir -p scripts

# Create Arduino ultrasonic sensor code
cat > scripts/arduino_ultrasonic.ino << 'EOF'
/*
Arduino Code for Autonomous Vehicle
Controls motors and reads ultrasonic sensors
*/

// Pin definitions
const int TRIG_PIN_FRONT = 9;
const int ECHO_PIN_FRONT = 10;
const int TRIG_PIN_LEFT = 11;
const int ECHO_PIN_LEFT = 12;
const int TRIG_PIN_RIGHT = 13;
const int ECHO_PIN_RIGHT = 14;

// Motor control pins
const int MOTOR_LEFT_FORWARD = 5;
const int MOTOR_LEFT_BACKWARD = 6;
const int MOTOR_RIGHT_FORWARD = 7;
const int MOTOR_RIGHT_BACKWARD = 8;
const int SERVO_PIN = 3;

// Variables
float front_distance, left_distance, right_distance;
int left_speed, right_speed, servo_angle;
unsigned long last_heartbeat = 0;
const unsigned long HEARTBEAT_INTERVAL = 100;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Initialize ultrasonic sensor pins
  pinMode(TRIG_PIN_FRONT, OUTPUT);
  pinMode(ECHO_PIN_FRONT, INPUT);
  pinMode(TRIG_PIN_LEFT, OUTPUT);
  pinMode(ECHO_PIN_LEFT, INPUT);
  pinMode(TRIG_PIN_RIGHT, OUTPUT);
  pinMode(ECHO_PIN_RIGHT, INPUT);
  
  // Initialize motor pins
  pinMode(MOTOR_LEFT_FORWARD, OUTPUT);
  pinMode(MOTOR_LEFT_BACKWARD, OUTPUT);
  pinMode(MOTOR_RIGHT_FORWARD, OUTPUT);
  pinMode(MOTOR_RIGHT_BACKWARD, OUTPUT);
  
  // Initialize servo
  pinMode(SERVO_PIN, OUTPUT);
  
  Serial.println("Arduino initialized");
}

void loop() {
  // Read ultrasonic sensors
  front_distance = readUltrasonic(TRIG_PIN_FRONT, ECHO_PIN_FRONT);
  left_distance = readUltrasonic(TRIG_PIN_LEFT, ECHO_PIN_LEFT);
  right_distance = readUltrasonic(TRIG_PIN_RIGHT, ECHO_PIN_RIGHT);
  
  // Send sensor data
  sendSensorData();
  
  // Check for motor commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    parseCommand(command);
  }
  
  // Send heartbeat
  if (millis() - last_heartbeat > HEARTBEAT_INTERVAL) {
    Serial.println("HEARTBEAT");
    last_heartbeat = millis();
  }
  
  delay(50);
}

float readUltrasonic(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  long duration = pulseIn(echoPin, HIGH);
  float distance = duration * 0.034 / 2;
  
  return distance;
}

void sendSensorData() {
  Serial.print("SENSOR:");
  Serial.print(front_distance);
  Serial.print(",");
  Serial.print(left_distance);
  Serial.print(",");
  Serial.println(right_distance);
}

void parseCommand(String command) {
  if (command.startsWith("MOTOR:")) {
    // Format: MOTOR:left_speed,right_speed,servo_angle
    command = command.substring(6);
    int comma1 = command.indexOf(',');
    int comma2 = command.indexOf(',', comma1 + 1);
    
    if (comma1 != -1 && comma2 != -1) {
      left_speed = command.substring(0, comma1).toInt();
      right_speed = command.substring(comma1 + 1, comma2).toInt();
      servo_angle = command.substring(comma2 + 1).toInt();
      
      setMotors(left_speed, right_speed);
      setServo(servo_angle);
      
      Serial.println("OK");
    }
  } else if (command == "STOP") {
    setMotors(0, 0);
    Serial.println("STOPPED");
  }
}

void setMotors(int left, int right) {
  // Left motor
  if (left > 0) {
    analogWrite(MOTOR_LEFT_FORWARD, abs(left));
    analogWrite(MOTOR_LEFT_BACKWARD, 0);
  } else {
    analogWrite(MOTOR_LEFT_FORWARD, 0);
    analogWrite(MOTOR_LEFT_BACKWARD, abs(left));
  }
  
  // Right motor
  if (right > 0) {
    analogWrite(MOTOR_RIGHT_FORWARD, abs(right));
    analogWrite(MOTOR_RIGHT_BACKWARD, 0);
  } else {
    analogWrite(MOTOR_RIGHT_FORWARD, 0);
    analogWrite(MOTOR_RIGHT_BACKWARD, abs(right));
  }
}

void setServo(int angle) {
  // Convert angle to pulse width
  int pulse = map(angle, 0, 180, 544, 2400);
  analogWrite(SERVO_PIN, pulse);
}
EOF
```

---

## Testing and Validation

### Step 1: Test Individual Components
```bash
# Test camera
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print('Camera working')
    cv2.imwrite('test_camera.jpg', frame)
else:
    print('Camera not working')
cap.release()
"

# Test Arduino communication
python3 -c "
import serial
try:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    ser.write(b'HEARTBEAT\n')
    response = ser.readline().decode().strip()
    print(f'Arduino response: {response}')
    ser.close()
except Exception as e:
    print(f'Arduino error: {e}')
"

# Test YOLO model
python3 -c "
from ultralytics import YOLO
import numpy as np
model = YOLO('yolov8n.pt')
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
results = model(test_image)
print(f'YOLO working: {len(results)} results')
"
```

### Step 2: Run System Tests
```bash
# Run simple tests
python3 test_simple.py

# Run quick tests
python3 quick_test.py

# Run LSTM tests
python3 test_lstm.py

# Run motor tests
python3 test_motors.py
```

### Step 3: Test Full System
```bash
# Test autonomous operation
python3 main.py --test

# Test demo mode
python3 main.py --demo

# Test normal operation
python3 main.py
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Camera Not Working
```bash
# Check camera module
vcgencmd get_camera

# Enable camera in raspi-config
sudo raspi-config

# Check camera permissions
ls -l /dev/video*

# Test with raspistill
raspistill -o test.jpg
```

#### 2. Arduino Not Detected
```bash
# Check available ports
ls /dev/tty*

# Check Arduino permissions
sudo chmod 666 /dev/ttyUSB0

# Test Arduino IDE connection
# Upload the arduino_ultrasonic.ino sketch
```

#### 3. YOLO Model Issues
```bash
# Reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics

# Download model manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

#### 4. Performance Issues
```bash
# Check system resources
htop

# Monitor temperature
vcgencmd measure_temp

# Check memory usage
free -h

# Optimize for performance
echo 'gpu_mem=128' | sudo tee -a /boot/config.txt
```

#### 5. Network Issues
```bash
# Check WiFi connection
iwconfig

# Check IP address
hostname -I

# Test internet connection
ping -c 3 google.com
```

---

## System Verification Checklist

### Hardware Setup
- [ ] Raspberry Pi 4 powered and connected
- [ ] Camera module connected and working
- [ ] Arduino connected via USB
- [ ] Motors connected to motor driver
- [ ] Ultrasonic sensors connected
- [ ] Power supply adequate

### Software Setup
- [ ] Raspberry Pi OS installed and updated
- [ ] Python 3 and pip installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Project files copied to Raspberry Pi
- [ ] Configuration files created

### Component Testing
- [ ] Camera captures images
- [ ] Arduino responds to commands
- [ ] YOLO model loads and runs
- [ ] Path planning algorithm works
- [ ] LSTM behavior prediction functions
- [ ] Motor control commands sent

### Integration Testing
- [ ] Object detection working
- [ ] Sensor data being read
- [ ] Path planning generating routes
- [ ] Motor commands being executed
- [ ] System logging operational
- [ ] Emergency stop functional

---

## Next Steps

1. **Hardware Assembly**: Mount all components on the vehicle chassis
2. **Calibration**: Calibrate sensors and motors
3. **Testing**: Perform real-world testing in controlled environment
4. **Optimization**: Fine-tune parameters for better performance
5. **Documentation**: Record test results and system behavior
6. **Presentation**: Prepare for graduation project demonstration

---

## Support and Resources

- **Raspberry Pi Documentation**: [raspberrypi.org/documentation](https://www.raspberrypi.org/documentation/)
- **Arduino Documentation**: [arduino.cc/reference](https://www.arduino.cc/reference/)
- **OpenCV Documentation**: [docs.opencv.org](https://docs.opencv.org/)
- **PyTorch Documentation**: [pytorch.org/docs](https://pytorch.org/docs/)
- **Ultralytics Documentation**: [docs.ultralytics.com](https://docs.ultralytics.com/)

---

*This guide provides a complete setup process for the AI-Based Autonomous Vehicle system. Follow each step carefully and test components as you go to ensure a successful implementation.* 