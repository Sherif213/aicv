#!/bin/bash

# AI-Based Autonomous Vehicle - Installation Script
# This script automates the installation process for the autonomous vehicle system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.8"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python $PYTHON_VERSION found, but Python 3.8+ is required"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher"
        return 1
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux (Ubuntu/Debian)
        if command_exists apt-get; then
            sudo apt-get update
            sudo apt-get install -y \
                python3-pip \
                python3-venv \
                python3-dev \
                libatlas-base-dev \
                libhdf5-dev \
                libhdf5-serial-dev \
                libatlas-base-dev \
                libjasper-dev \
                libqtcore4 \
                libqtgui4 \
                libqt4-test \
                libgstreamer1.0-0 \
                libgstreamer-plugins-base1.0-0 \
                libgtk-3-0 \
                libavcodec-dev \
                libavformat-dev \
                libswscale-dev \
                libv4l-dev \
                libxvidcore-dev \
                libx264-dev \
                libjpeg-dev \
                libpng-dev \
                libtiff-dev \
                libatlas-base-dev \
                gfortran \
                wget \
                curl \
                git
        elif command_exists yum; then
            # CentOS/RHEL
            sudo yum update -y
            sudo yum install -y \
                python3-pip \
                python3-devel \
                atlas-devel \
                hdf5-devel \
                gcc \
                gcc-c++ \
                make \
                wget \
                curl \
                git
        else
            print_warning "Unsupported Linux distribution. Please install dependencies manually."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            brew update
            brew install \
                python3 \
                opencv \
                numpy \
                wget \
                curl \
                git
        else
            print_warning "Homebrew not found. Please install Homebrew first: https://brew.sh/"
        fi
    else
        print_warning "Unsupported operating system: $OSTYPE"
    fi
}

# Function to create virtual environment
create_virtual_environment() {
    print_status "Creating Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
}

# Function to install Python dependencies
install_python_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Function to download YOLO model
download_yolo_model() {
    print_status "Downloading YOLO model..."
    
    # Create models directory if it doesn't exist
    mkdir -p models
    
    # Download YOLOv8n model if it doesn't exist
    if [ ! -f "models/yolov8n.pt" ]; then
        wget -O models/yolov8n.pt \
            https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
        
        if [ $? -eq 0 ]; then
            print_success "YOLO model downloaded successfully"
        else
            print_error "Failed to download YOLO model"
            return 1
        fi
    else
        print_status "YOLO model already exists"
    fi
}

# Function to setup Arduino
setup_arduino() {
    print_status "Setting up Arduino..."
    
    if command_exists arduino-cli; then
        print_status "Arduino CLI found"
        
        # Install required libraries
        arduino-cli lib install "RunningAverage"
        arduino-cli lib install "Servo"
        
        print_success "Arduino libraries installed"
    else
        print_warning "Arduino CLI not found. Please install Arduino IDE and upload the code manually:"
        print_warning "1. Open scripts/arduino_ultrasonic.ino in Arduino IDE"
        print_warning "2. Install RunningAverage and Servo libraries"
        print_warning "3. Upload to your Arduino board"
    fi
}

# Function to setup permissions
setup_permissions() {
    print_status "Setting up permissions..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Add user to dialout group for serial access
        if ! groups $USER | grep -q dialout; then
            sudo usermod -a -G dialout $USER
            print_warning "Added user to dialout group. Please reboot for changes to take effect."
        fi
        
        # Add user to video group for camera access
        if ! groups $USER | grep -q video; then
            sudo usermod -a -G video $USER
        fi
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating project directories..."
    
    mkdir -p data
    mkdir -p logs
    mkdir -p exports
    
    print_success "Directories created"
}

# Function to run system tests
run_tests() {
    print_status "Running system tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run basic tests
    python -c "
import sys
import cv2
import numpy as np
print('✓ OpenCV:', cv2.__version__)
print('✓ NumPy:', np.__version__)
print('✓ Python:', sys.version)
"
    
    print_success "Basic tests passed"
}

# Function to display next steps
display_next_steps() {
    echo
    print_success "Installation completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Connect your Arduino and upload the code from scripts/arduino_ultrasonic.ino"
    echo "2. Connect your camera and sensors"
    echo "3. Update hardware configuration in config/settings.py"
    echo "4. Run system tests: python scripts/test_system.py"
    echo "5. Start the system: python main.py"
    echo
    echo "For more information, see README.md"
    echo
}

# Main installation function
main() {
    echo "AI-Based Autonomous Vehicle - Installation Script"
    echo "=================================================="
    echo
    
    # Check Python version
    if ! check_python_version; then
        exit 1
    fi
    
    # Install system dependencies
    install_system_dependencies
    
    # Create virtual environment
    create_virtual_environment
    
    # Install Python dependencies
    install_python_dependencies
    
    # Download YOLO model
    download_yolo_model
    
    # Setup Arduino
    setup_arduino
    
    # Setup permissions
    setup_permissions
    
    # Create directories
    create_directories
    
    # Run tests
    run_tests
    
    # Display next steps
    display_next_steps
}

# Check if script is run with sudo
if [[ $EUID -eq 0 ]]; then
    print_error "This script should not be run as root"
    exit 1
fi

# Run main function
main "$@" 