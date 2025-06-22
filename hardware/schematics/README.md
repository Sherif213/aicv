# Hardware Schematics - AI-Based Autonomous Vehicle

This directory contains circuit diagrams and wiring schematics for the autonomous vehicle system.

## 📋 Schematic Files

### 1. Main System Architecture
- **File**: `system_architecture.pdf`
- **Description**: Overall system block diagram showing connections between Raspberry Pi, Arduino, sensors, and actuators
- **Components**: All major system components and their interconnections

### 2. Arduino Circuit
- **File**: `arduino_circuit.pdf`
- **Description**: Detailed Arduino wiring diagram
- **Components**: 
  - Arduino Uno
  - Motor driver (L298N)
  - Ultrasonic sensors (HC-SR04 x3)
  - Servo motor
  - Power distribution

### 3. Motor Control Circuit
- **File**: `motor_control.pdf`
- **Description**: Motor driver and motor connections
- **Components**:
  - L298N motor driver
  - DC motors (2x)
  - Power supply connections
  - Control signal routing

### 4. Sensor Wiring
- **File**: `sensor_wiring.pdf`
- **Description**: Sensor connections and power distribution
- **Components**:
  - Ultrasonic sensors
  - Camera module
  - IMU sensor (optional)
  - Power and signal connections

## 🔌 Pin Assignments

### Arduino Pin Mapping

| Pin | Function | Component | Notes |
|-----|----------|-----------|-------|
| 5 | PWM | Motor Left Forward | Motor control |
| 6 | PWM | Motor Left Backward | Motor control |
| 7 | PWM | Motor Right Forward | Motor control |
| 8 | PWM | Motor Right Backward | Motor control |
| 9 | Digital | Ultrasonic Front Trigger | Distance sensor |
| 10 | Digital | Ultrasonic Front Echo | Distance sensor |
| 11 | Digital | Ultrasonic Left Trigger | Distance sensor |
| 12 | Digital | Ultrasonic Left Echo | Distance sensor |
| 13 | Digital | Ultrasonic Right Trigger | Distance sensor |
| 14 | Digital | Ultrasonic Right Echo | Distance sensor |
| 3 | PWM | Servo Motor | Steering control |
| A4 | I2C SDA | IMU Sensor | Orientation (optional) |
| A5 | I2C SCL | IMU Sensor | Orientation (optional) |

### Raspberry Pi Pin Mapping

| Pin | Function | Component | Notes |
|-----|----------|-----------|-------|
| GPIO 17 | Digital | Motor Left Forward | Backup control |
| GPIO 18 | Digital | Motor Left Backward | Backup control |
| GPIO 27 | Digital | Motor Right Forward | Backup control |
| GPIO 22 | Digital | Motor Right Backward | Backup control |
| GPIO 23 | PWM | Servo Motor | Backup steering |
| GPIO 24 | Digital | Ultrasonic Trigger | Backup sensor |
| GPIO 25 | Digital | Ultrasonic Echo | Backup sensor |
| I2C1 SDA | I2C | IMU Sensor | Orientation data |
| I2C1 SCL | I2C | IMU Sensor | Orientation data |

## 🔋 Power Distribution

### Power Requirements

| Component | Voltage | Current | Power |
|-----------|---------|---------|-------|
| Raspberry Pi 4 | 5V | 3A | 15W |
| Arduino Uno | 5V | 0.5A | 2.5W |
| DC Motors (2x) | 12V | 2A | 24W |
| Servo Motor | 5V | 0.5A | 2.5W |
| Sensors | 5V | 0.2A | 1W |
| **Total** | - | - | **45W** |

### Power Supply Configuration

```
12V Battery Pack
├── 12V Rail
│   ├── Motor Driver
│   └── DC Motors
└── 5V Regulator
    ├── Raspberry Pi
    ├── Arduino
    ├── Servo Motor
    └── Sensors
```

## 🛠️ Assembly Instructions

### Step 1: Arduino Assembly
1. Connect motor driver to Arduino
2. Wire ultrasonic sensors
3. Connect servo motor
4. Test individual components

### Step 2: Raspberry Pi Assembly
1. Connect camera module
2. Wire backup control pins
3. Connect I2C sensors
4. Test communication

### Step 3: Power Distribution
1. Connect 12V battery to motor driver
2. Connect 5V regulator to Pi and Arduino
3. Add power switches and fuses
4. Test power distribution

### Step 4: Integration
1. Connect Arduino to Raspberry Pi via USB
2. Mount all components on chassis
3. Route and secure all cables
4. Perform system integration test

## ⚠️ Safety Considerations

### Electrical Safety
- Use appropriate wire gauges for current requirements
- Add fuses for overcurrent protection
- Ensure proper grounding
- Use heat shrink tubing for wire insulation

### Mechanical Safety
- Secure all components firmly to chassis
- Protect sensitive electronics from vibration
- Ensure proper cable routing to prevent damage
- Add emergency stop functionality

### Operational Safety
- Test all systems before autonomous operation
- Implement fail-safe mechanisms
- Monitor system temperature and power consumption
- Have manual override capabilities

## 🔧 Troubleshooting

### Common Issues
1. **Motor not responding**: Check motor driver connections and power supply
2. **Sensors not working**: Verify power and signal connections
3. **Communication errors**: Check USB cable and port settings
4. **Power issues**: Monitor voltage levels and current draw

### Testing Procedures
1. **Continuity test**: Verify all electrical connections
2. **Voltage test**: Check power supply levels
3. **Signal test**: Verify control signals reach components
4. **Integration test**: Test complete system functionality

## 📐 Mechanical Layout

### Chassis Design Considerations
- **Size**: Accommodate all components with room for maintenance
- **Weight**: Balance between strength and weight
- **Accessibility**: Easy access to components for testing and repair
- **Modularity**: Design for easy component replacement

### Component Placement
- **Raspberry Pi**: Central location with good ventilation
- **Arduino**: Near motor driver for short control wires
- **Batteries**: Low center of gravity for stability
- **Sensors**: Unobstructed view for optimal performance

---

*These schematics provide the foundation for building a reliable and safe autonomous vehicle system. Always follow proper safety procedures and test thoroughly before operation.* 