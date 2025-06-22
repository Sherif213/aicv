# AI-Based Autonomous Vehicle for Commercial Use

## 🚗 Project Overview

This project implements a complete autonomous vehicle system using Raspberry Pi, Arduino, and AI algorithms. The system performs real-time object detection, path planning, obstacle avoidance, and behavior prediction for safe autonomous navigation.

## 📁 Project Structure

```
aicv/
├── 📁 core/                    # Core system modules
│   ├── arduino_communication.py
│   ├── logger.py
│   ├── path_planning.py
│   └── vehicle_controller.py
├── 📁 models/                  # AI/ML models and detection
│   ├── object_detection.py
│   ├── behavior_prediction.py
│   └── yolov8n.pt
├── 📁 config/                  # Configuration files
│   └── settings.py
├── 📁 tests/                   # Test scripts
│   ├── test_simple.py
│   ├── test_lstm.py
│   ├── test_motors.py
│   ├── quick_test.py
│   └── move_car.py
├── 📁 scripts/                 # Utility scripts
│   ├── 📁 arduino/            # Arduino firmware
│   │   └── arduino_ultrasonic.ino
│   ├── 📁 install/            # Installation scripts
│   │   └── install.sh
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
├── main.py                    # Main application entry point
└── venv/                     # Python virtual environment
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download the project
cd aicv

# Run installation script
chmod +x scripts/install/install.sh
./scripts/install/install.sh
```

### 2. Hardware Setup
1. Connect Arduino to Raspberry Pi via USB
2. Connect camera module to Raspberry Pi
3. Wire motors and sensors according to schematics
4. Upload Arduino firmware: `scripts/arduino/arduino_ultrasonic.ino`

### 3. Testing
```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
python tests/test_simple.py      # Basic functionality
python tests/test_lstm.py        # LSTM behavior prediction
python tests/test_motors.py      # Motor control
python tests/quick_test.py       # Quick system test
```

### 4. Running the System
```bash
# Test mode
python main.py --test

# Demo mode (no hardware required)
python main.py --demo

# Full autonomous operation
python main.py
```

## 🔧 System Components

### AI/ML Modules
- **Object Detection**: YOLOv8 real-time detection
- **Path Planning**: A* algorithm for optimal navigation
- **Behavior Prediction**: LSTM networks for trajectory forecasting
- **Sensor Fusion**: Multi-sensor data integration

### Hardware Integration
- **Raspberry Pi 4**: Central processing unit
- **Arduino**: Motor control and sensor reading
- **Camera**: Real-time vision processing
- **Ultrasonic Sensors**: Obstacle detection
- **DC Motors**: Vehicle propulsion
- **Servo Motor**: Steering control

### Software Architecture
- **Modular Design**: Separate modules for each component
- **Real-time Processing**: 20Hz control loop
- **Comprehensive Logging**: Performance and system monitoring
- **Error Handling**: Robust error recovery and emergency stops

## 📊 Performance Metrics

- **Object Detection**: 30+ FPS with YOLOv8
- **Path Planning**: <100ms response time
- **Behavior Prediction**: LSTM inference <50ms
- **Control Loop**: 20Hz stable operation
- **Emergency Stop**: <50ms response time

## 🛠️ Configuration

Edit `config/settings.py` to customize:
- Hardware pin assignments
- AI model parameters
- Control system settings
- Communication protocols
- Logging preferences

## 📚 Documentation

- **[Project Summary](PROJECT_SUMMARY.md)**: Complete project overview
- **[Raspberry Pi Setup](RASPBERRY_PI_SETUP_GUIDE.md)**: Detailed setup instructions
- **[Hardware Schematics](hardware/schematics/)**: Circuit diagrams
- **[Bill of Materials](hardware/bom/)**: Component list

## 🧪 Testing

### Automated Tests
```bash
# Run all tests
python tests/test_simple.py
python tests/test_lstm.py
python tests/test_motors.py
python tests/quick_test.py

# Test specific components
python tests/move_car.py
```

### Manual Testing
- Camera functionality
- Motor control response
- Sensor data accuracy
- Path planning algorithms
- Emergency stop systems

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

## 📈 Future Enhancements

- **Advanced SLAM**: Simultaneous Localization and Mapping
- **Multi-vehicle Coordination**: Fleet management capabilities
- **Cloud Integration**: Remote monitoring and control
- **Advanced Sensors**: LiDAR integration for 3D perception
- **Machine Learning**: Continuous learning from operation data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is developed for academic purposes as a graduation project.

## 👨‍🎓 Academic Context

**Project Title**: AI-Based Autonomous Vehicles for Commercial Use  
**Institution**: [Your University]  
**Department**: [Your Department]  
**Supervisor**: [Your Supervisor]  
**Year**: 2024

---

*For detailed setup instructions, see [RASPBERRY_PI_SETUP_GUIDE.md](RASPBERRY_PI_SETUP_GUIDE.md)*
