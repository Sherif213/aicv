# Video Recording Guide for AI Autonomous Vehicle Project

## Overview
This guide will help you create a professional 3-minute demonstration video showcasing your AI-based autonomous vehicle project.

## Pre-Recording Checklist

### 1. System Preparation
- [ ] Ensure all Python dependencies are installed
- [ ] Test the system with `python scripts/test_demo.py`
- [ ] Verify camera access (if available)
- [ ] Check Arduino connection (if hardware is available)
- [ ] Prepare a clean desktop/workspace for recording

### 2. Recording Setup
- [ ] Use screen recording software (OBS Studio, SimpleScreenRecorder, or built-in tools)
- [ ] Set recording resolution to 1920x1080 or higher
- [ ] Test audio recording (optional narration)
- [ ] Prepare a script or outline for the demonstration

## Video Structure (3 Minutes)

### Introduction (30 seconds)
- Project title: "AI-Based Autonomous Vehicle for Commercial Use"
- Brief overview of the system components
- Show the project structure and codebase

### System Components (1 minute)
1. **Object Detection with YOLOv8** (20 seconds)
   - Show real-time object detection
   - Display bounding boxes and confidence scores
   - Mention the classes detected (person, car, truck, etc.)

2. **Path Planning with A* Algorithm** (20 seconds)
   - Show path planning visualization
   - Demonstrate obstacle avoidance
   - Display the grid-based navigation system

3. **Behavior Prediction with LSTM** (20 seconds)
   - Show behavior prediction in action
   - Display confidence scores
   - Explain the prediction features

### Hardware Integration (30 seconds)
- **Arduino Communication**
  - Show motor control commands
  - Display ultrasonic sensor readings
  - Demonstrate real-time communication

### Full System Demo (1 minute)
- **Complete Autonomous Navigation**
  - Show the vehicle navigating to multiple goals
  - Display real-time status updates
  - Demonstrate obstacle avoidance
  - Show system health monitoring

## Recording Commands

### 1. Test the System First
```bash
cd /home/tayqon/aicv
python scripts/test_demo.py
```

### 2. Run the Full Demonstration
```bash
python scripts/demo_video.py
```

### 3. Alternative: Run Individual Components
```bash
# Test object detection
python main.py --test

# Run a simple demo
python main.py --demo

# Run with specific goal
python main.py --goal 50 50
```

## Recording Tips

### 1. Screen Recording
- Use OBS Studio for professional quality
- Set frame rate to 30 FPS
- Enable hardware acceleration if available
- Record in 1080p resolution

### 2. Audio (Optional)
- Use a good microphone for narration
- Speak clearly and at a moderate pace
- Explain technical concepts in simple terms
- Keep background noise minimal

### 3. Visual Presentation
- Use a dark theme terminal for better visibility
- Zoom in on important code sections
- Show file structure and project organization
- Highlight key features and results

## Post-Recording

### 1. Video Editing
- Trim unnecessary parts
- Add captions for technical terms
- Include project title and credits
- Add smooth transitions between sections

### 2. File Management
- Save the video as `autonomous_vehicle_demo.mp4`
- Keep file size under 100MB for easy sharing
- Create a backup copy

## Troubleshooting

### Common Issues
1. **Camera not working**: The demo will use simulation mode
2. **Arduino not connected**: The demo will show simulated communication
3. **Model loading errors**: Check if YOLOv8 model file exists
4. **Import errors**: Ensure all dependencies are installed

### Fallback Options
- Use simulation mode for hardware components
- Show code structure and documentation
- Demonstrate individual components separately
- Use screenshots of successful runs

## Video Script Template

### Opening (30 seconds)
"Welcome to the AI-Based Autonomous Vehicle project demonstration. This system integrates computer vision, path planning, and machine learning to create a fully autonomous vehicle capable of real-time navigation and obstacle avoidance."

### Main Content (2 minutes)
"Let me show you the key components: First, our YOLOv8 object detection system identifies obstacles in real-time. Next, the A* path planning algorithm calculates optimal routes while avoiding obstacles. Our LSTM behavior prediction model analyzes vehicle patterns for safer navigation. Finally, Arduino communication enables precise motor control and sensor reading."

### Closing (30 seconds)
"This demonstration shows how AI algorithms work together to create a robust autonomous vehicle system. The project successfully integrates computer vision, path planning, and machine learning for commercial applications."

## Final Notes
- Keep the video focused and professional
- Highlight the technical achievements
- Show both code and running demonstrations
- Emphasize the integration of multiple AI components
- End with a clear summary of the project's capabilities 