# AI-Based Autonomous Vehicle for Commercial Use




## 🚗 Project Overview

This project implements a complete autonomous vehicle system using Raspberry Pi, Arduino, and advanced AI algorithms. The system performs real-time object detection, intelligent path planning, obstacle avoidance, and behavior prediction for safe autonomous navigation.

### Key Features
- **Real-time Object Detection** using YOLOv8
- **A* Path Planning** for optimal navigation
- **LSTM Behavior Prediction** for trajectory forecasting
- **Multi-sensor Fusion** (Camera + Ultrasonic + IMU)
- **Robust Motor Control** via Arduino
- **Comprehensive Safety Systems**
- **Professional Logging and Monitoring**

---

## 📁 Project Structure

```
aicv/
├── 📁 core/                    # Core system modules
│   ├── arduino_communication.py
│   ├── logger.py
│   ├── path_planning.py
│   └── vehicle_controller.py
├── 📁 models/                  # AI/ML models
│   ├── object_detection.py
│   ├── behavior_prediction.py
│   └── yolov8n.pt
├── 📁 config/                  # Configuration
│   └── settings.py
├── 📁 tests/                   # Test scripts
│   ├── test_simple.py
│   ├── test_lstm.py
│   ├── test_motors.py
│   ├── quick_test.py
│   └── move_car.py
├── 📁 scripts/                 # Utility scripts
│   ├── 📁 arduino/            # Arduino firmware
│   ├── 📁 install/            # Installation scripts
│   └── test_system.py
├── 📁 docs/                    # Documentation
│   ├── README.md
│   ├── PROJECT_SUMMARY.md
│   ├── RASPBERRY_PI_SETUP_GUIDE.md
│   └── requirements.txt
├── 📁 hardware/               # Hardware documentation
│   ├── 📁 schematics/         # Circuit diagrams
│   └── 📁 bom/               # Bill of materials
├── 📁 logs/                   # System logs
├── 📁 data/                   # Data storage
├── main.py                    # Main application
└── venv/                     # Python environment
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download the project
cd aicv

# Run installation script
chmod +x scripts/install/install.sh
./scripts/install/install.sh
```

### 2. Testing
```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
python tests/test_simple.py      # Basic functionality
python tests/test_lstm.py        # LSTM behavior prediction
python tests/test_motors.py      # Motor control
python tests/quick_test.py       # Quick system test
```

### 3. Running the System
```bash
# Test mode
python main.py --test

# Demo mode (no hardware required)
python main.py --demo

# Full autonomous operation
python main.py
```

---

## 📚 Documentation

### 📖 [Complete Documentation](docs/README.md)
Detailed project documentation including setup, usage, and troubleshooting.

### 🖥️ [Raspberry Pi Setup Guide](docs/RASPBERRY_PI_SETUP_GUIDE.md)
Step-by-step guide from initial imaging to full system operation.

### 📋 [Project Summary](docs/PROJECT_SUMMARY.md)
Comprehensive overview of objectives, methodology, and results.

### 🔧 [Hardware Documentation](hardware/)
- [Circuit Schematics](hardware/schematics/)

---

## 🔧 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raspberry Pi  │    │     Arduino     │    │     Sensors     │
│                 │    │                 │    │                 │
│ • YOLO Model    │◄──►│ • Motor Control │◄──►│ • Ultrasonic    │
│ • Path Planning │    │ • Servo Control │    │ • Camera        │
│ • LSTM Predict  │    │ • Sensor Read   │    │ • IMU           │
│ • Main Control  │    │ • Communication │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 📊 Performance Metrics

- **Object Detection**: 30+ FPS with YOLOv8
- **Path Planning**: <100ms response time
- **Behavior Prediction**: LSTM inference <50ms
- **Control Loop**: 20Hz stable operation
- **Emergency Stop**: <50ms response time

---

## 🛠️ Hardware Requirements

### Essential Components
- **Raspberry Pi 4** (4GB+ RAM)
- **Arduino Uno/Mega**
- **Camera Module** (Pi Camera v2 or USB)
- **Ultrasonic Sensors** (HC-SR04 x3)
- **DC Motors** (2x) with motor driver
- **Servo Motor** for steering
- **Chassis and Wheels**

### Optional Components
- **LiDAR Sensor** for 3D perception
- **GPS Module** for global positioning
- **IMU Sensor** for orientation
- **Display** for status monitoring

---

## 🧪 Testing

### Automated Tests
```bash
# Run all tests
python tests/test_simple.py
python tests/test_lstm.py
python tests/test_motors.py
python tests/quick_test.py
```

### Manual Testing
- Camera functionality
- Motor control response
- Sensor data accuracy
- Path planning algorithms
- Emergency stop systems

---

## 🔍 Troubleshooting

### Common Issues
1. **Camera not detected**: Check ribbon cable and enable in raspi-config
2. **Arduino not responding**: Verify USB connection and port settings
3. **YOLO model errors**: Reinstall ultralytics and download model
4. **Performance issues**: Monitor system resources and temperature

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py --test
```

---

## 📈 Academic Contributions

### Technical Innovations
- **LSTM-based Behavior Prediction**: Advanced trajectory forecasting
- **Multi-sensor Fusion**: Robust perception system
- **Real-time A* Path Planning**: Efficient navigation algorithms
- **Modular Architecture**: Scalable and maintainable design

### Research Impact
- Demonstrates practical AI implementation in autonomous systems
- Provides framework for commercial autonomous vehicle development
- Contributes to open-source autonomous vehicle community
- Advances understanding of real-time AI systems

