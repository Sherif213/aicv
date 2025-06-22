/*
 * Arduino Code for AI-Based Autonomous Vehicle
 * Controls ultrasonic sensors and motors
 * Communicates with Raspberry Pi via serial
 */

#include <RunningAverage.h>
#include <Servo.h>

// Pin Definitions
const byte SENSOR_COUNT = 3;
const byte TRIG_PINS[SENSOR_COUNT] = {3, 7, 6};  // Front, Left, Right  
const byte ECHO_PINS[SENSOR_COUNT] = {2, 5, 4};

// Motor control pins
const byte MOTOR_LEFT_FORWARD = 8;
const byte MOTOR_LEFT_BACKWARD = 9;
const byte MOTOR_RIGHT_FORWARD = 10;
const byte MOTOR_RIGHT_BACKWARD = 11;
const byte SERVO_STEERING_PIN = 12;

// Timing Constants 
const unsigned long PI_SERIAL_DELAY = 3000;  // 3s for Pi serial initialization
const unsigned long SAMPLE_INTERVAL = 100;   // ms (10Hz)
const unsigned long PULSE_TIMEOUT = 25000;   // μs (4m max)

// Safety Thresholds (cm)
const int WARNING_DIST = 100;
const int CRITICAL_DIST = 40;
const int MIN_DIST = 2;

// Filtering
RunningAverage filters[SENSOR_COUNT] = {
  RunningAverage(5), RunningAverage(5), RunningAverage(5)
};

// Servo for steering
Servo steeringServo;

// Motor control variables
int leftSpeed = 0;
int rightSpeed = 0;
int steeringAngle = 90;  // 90 = straight

// Communication variables
String inputString = "";
boolean stringComplete = false;

void setup() {
  // Extended initialization for Pi
  delay(PI_SERIAL_DELAY);
  Serial.begin(115200);
  while (!Serial && millis() < 5000);  // Wait for serial

  // Initialize sensors
  for (byte i = 0; i < SENSOR_COUNT; i++) {
    pinMode(TRIG_PINS[i], OUTPUT);
    pinMode(ECHO_PINS[i], INPUT);
    digitalWrite(TRIG_PINS[i], LOW);
  }
  
  // Initialize motor pins
  pinMode(MOTOR_LEFT_FORWARD, OUTPUT);
  pinMode(MOTOR_LEFT_BACKWARD, OUTPUT);
  pinMode(MOTOR_RIGHT_FORWARD, OUTPUT);
  pinMode(MOTOR_RIGHT_BACKWARD, OUTPUT);
  
  // Initialize servo
  steeringServo.attach(SERVO_STEERING_PIN);
  steeringServo.write(90);  // Center position
  
  // Stop motors initially
  stopMotors();
  
  // Pi-friendly startup message
  Serial.println("RPI_ULTRASONIC_STARTED");
}

void loop() {
  static unsigned long lastSample = 0;
  
  // Handle serial communication
  if (stringComplete) {
    parseCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  // Sensor sampling
  if (millis() - lastSample >= SAMPLE_INTERVAL) {
    lastSample = millis();
    
    // Read ultrasonic sensors
    float leftDist = getFilteredDistance(0);
    float frontDist = getFilteredDistance(1);
    float rightDist = getFilteredDistance(2);
    
    // Pi-optimized output format:
    // L:distance F:distance R:distance
    Serial.print("L:");
    Serial.print(leftDist);
    Serial.print(" F:");
    Serial.print(frontDist);
    Serial.print(" R:");
    Serial.println(rightDist);
    
    // Send motor status
    sendMotorStatus();
  }
}

float getFilteredDistance(byte sensorIndex) {
  // Generate pulse
  digitalWrite(TRIG_PINS[sensorIndex], LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PINS[sensorIndex], HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PINS[sensorIndex], LOW);

  // Read echo with timeout
  long duration = pulseIn(ECHO_PINS[sensorIndex], HIGH, PULSE_TIMEOUT);
  
  // Calculate and validate distance
  if (duration <= 0) return -1.0;
  float distance = duration * 0.034 / 2;
  if (distance < MIN_DIST || distance > 400) return -1.0;
  
  // Apply filtering
  filters[sensorIndex].addValue(distance);
  return filters[sensorIndex].getAverage();
}

void parseCommand(String command) {
  command.trim();
  
  // Motor command: MOTOR:L:left_speed:R:right_speed
  if (command.startsWith("MOTOR:")) {
    parseMotorCommand(command);
  }
  // Servo command: SERVO:angle
  else if (command.startsWith("SERVO:")) {
    parseServoCommand(command);
  }
  // Emergency stop
  else if (command == "EMERGENCY_STOP") {
    emergencyStop();
  }
  // Heartbeat
  else if (command == "HEARTBEAT") {
    Serial.println("HEARTBEAT");
  }
}

void parseMotorCommand(String command) {
  // Format: MOTOR:L:left_speed:R:right_speed
  int lIndex = command.indexOf("L:");
  int rIndex = command.indexOf("R:");
  
  if (lIndex != -1 && rIndex != -1) {
    // Extract left speed
    int lEnd = command.indexOf(":", lIndex + 2);
    if (lEnd != -1) {
      String leftStr = command.substring(lIndex + 2, lEnd);
      leftSpeed = leftStr.toInt();
    }
    
    // Extract right speed
    String rightStr = command.substring(rIndex + 2);
    rightSpeed = rightStr.toInt();
    
    // Apply motor control
    setMotorSpeeds(leftSpeed, rightSpeed);
  }
}

void parseServoCommand(String command) {
  // Format: SERVO:angle
  int colonIndex = command.indexOf(":");
  if (colonIndex != -1) {
    String angleStr = command.substring(colonIndex + 1);
    steeringAngle = angleStr.toInt();
    
    // Clamp angle to valid range
    steeringAngle = constrain(steeringAngle, 0, 180);
    
    // Apply steering
    steeringServo.write(steeringAngle);
  }
}

void setMotorSpeeds(int left, int right) {
  // Clamp speeds to valid range
  left = constrain(left, -255, 255);
  right = constrain(right, -255, 255);
  
  // Set left motor
  if (left > 0) {
    analogWrite(MOTOR_LEFT_FORWARD, left);
    analogWrite(MOTOR_LEFT_BACKWARD, 0);
  } else {
    analogWrite(MOTOR_LEFT_FORWARD, 0);
    analogWrite(MOTOR_LEFT_BACKWARD, -left);
  }
  
  // Set right motor
  if (right > 0) {
    analogWrite(MOTOR_RIGHT_FORWARD, right);
    analogWrite(MOTOR_RIGHT_BACKWARD, 0);
  } else {
    analogWrite(MOTOR_RIGHT_FORWARD, 0);
    analogWrite(MOTOR_RIGHT_BACKWARD, -right);
  }
  
  leftSpeed = left;
  rightSpeed = right;
}

void stopMotors() {
  analogWrite(MOTOR_LEFT_FORWARD, 0);
  analogWrite(MOTOR_LEFT_BACKWARD, 0);
  analogWrite(MOTOR_RIGHT_FORWARD, 0);
  analogWrite(MOTOR_RIGHT_BACKWARD, 0);
  
  leftSpeed = 0;
  rightSpeed = 0;
}

void emergencyStop() {
  stopMotors();
  steeringServo.write(90);  // Center steering
  
  Serial.println("EMERGENCY_STOP_ACTIVATED");
}

void sendMotorStatus() {
  // Send current motor and servo status
  Serial.print("MOTOR_STATUS:L:");
  Serial.print(leftSpeed);
  Serial.print(":R:");
  Serial.print(rightSpeed);
  Serial.print(":S:");
  Serial.println(steeringAngle);
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
} 