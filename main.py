#!/usr/bin/env python3
"""
ESP32-CAM Fruit Spoilage Detector Main Entrypoint.
Launches the PyQt Glassmorphism AI Vision Application.
"""

import os
import sys

# Ensure src package is accessible on python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.app.gui_app import main

if __name__ == "__main__":
    main()
