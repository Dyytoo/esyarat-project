# Esyarat - AI-Assisted Sign Language Detection

Esyarat is an AI-powered application that helps people with disabilities by detecting sign language gestures through a camera and converting them into text.

## Features
- Real-time sign language detection using webcam
- Hand pose detection and tracking
- Neural network-based gesture recognition
- User-friendly web interface

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL
3. Allow camera access when prompted
4. Start making sign language gestures in front of the camera

## Project Structure
- `app.py`: Main Streamlit application
- `model/`: Contains the neural network model
- `utils/`: Utility functions for preprocessing and data handling
- `data/`: Directory for storing training data (to be added)

## Requirements
- Python 3.8+
- Webcam
- See requirements.txt for Python package dependencies 